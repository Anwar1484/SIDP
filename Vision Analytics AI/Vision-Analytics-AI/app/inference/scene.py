"""RelateAnything's published ONNX graph, with automatic object regions.

The graph and predicate bank are unchanged upstream artifacts. Preprocessing
and calibrated logit decoding follow the author's deployment score contract.
A capacity-one background worker keeps this slower scene pass off the live path.
"""
import json
import logging
import threading
from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from app.config import SceneConfig
from app.model_assets import model_manifest, verify_assets
from app.runtime import ROOT
from app.schemas import SceneEvidence

logger = logging.getLogger(__name__)
PREDICATES = (
    "sitting on", "sitting in", "sitting at", "standing beside", "standing near",
    "standing next to", "leaning against", "leaning on", "resting on", "resting against",
    "holding", "looking at", "using", "walking past", "sitting beside",
)
OBJECTS = {"chair", "couch", "bed", "bench", "dining table", "laptop", "tv", "cell phone",
           "book", "bottle", "cup", "handbag", "suitcase", "backpack", "keyboard"}


class SceneBackend:
    def __init__(self, config: SceneConfig):
        verify_assets("models/relateanything/", ROOT / config.directory)
        import onnxruntime as ort
        self.config = config
        directory = ROOT / config.directory
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = config.cpu_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.detector = ort.InferenceSession(str(directory / "yolo26n.onnx"), opts,
                                             providers=["CPUExecutionProvider"])
        self.relation = ort.InferenceSession(str(directory / "relateanything_vits16plus_w16.onnx"), opts,
                                             providers=["CPUExecutionProvider"])
        self.classes = json.loads((directory / "yolo26n.classes.json").read_text(encoding="utf-8"))
        if isinstance(self.classes, dict):
            self.classes = self.classes.get("classes", self.classes)
        bank = json.loads((directory / "predicate_bank.json").read_text(encoding="utf-8"))
        indices = [bank["names"].index(name) for name in PREDICATES]
        self.predicates = list(PREDICATES)
        self.W = np.ascontiguousarray(np.array(bank["W"], np.float32)[indices])
        self.alpha = np.ascontiguousarray(np.array(bank["alpha"], np.float32)[indices])
        self.meta = model_manifest()["relateanything"]
        self.detector_input = self.detector.get_inputs()[0].name

    def objects(self, frame: np.ndarray):
        height, width = frame.shape[:2]
        size = int(self.detector.get_inputs()[0].shape[-1])
        scale = min(size/width, size/height)
        nw, nh = round(width*scale), round(height*scale)
        px, py = (size-nw)//2, (size-nh)//2
        padded = np.full((size, size, 3), 114, np.uint8)
        padded[py:py+nh, px:px+nw] = cv2.resize(frame, (nw, nh))
        tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], np.float32)/255
        rows = self.detector.run(None, {self.detector_input: tensor})[0][0]
        found = []
        for row in rows:
            x1, y1, x2, y2, confidence, category = row[:6]
            label = self.classes[int(category)] if isinstance(self.classes, list) else self.classes[str(int(category))]
            if confidence < .25 or label not in OBJECTS:
                continue
            box = np.array([(x1-px)/scale, (y1-py)/scale, (x2-px)/scale, (y2-py)/scale], np.float32)
            box[[0, 2]] = box[[0, 2]].clip(0, width)
            box[[1, 3]] = box[[1, 3]].clip(0, height)
            if box[2] > box[0] and box[3] > box[1]:
                found.append((box, label, float(confidence)))
        return sorted(found, key=lambda entry: entry[2], reverse=True)[:20]

    def infer(self, frame: np.ndarray, people: list[tuple[int, list[float]]], timestamp: float):
        start = perf_counter()
        output = {identity: SceneEvidence(status="no_relation", sampled_at=timestamp, age_seconds=0)
                  for identity, _ in people}
        objects = self.objects(frame)
        if not objects or not people:
            return output
        people = people[:12]
        boxes = np.array([box for _, box in people] + [o[0] for o in objects], np.float32)
        labels = ["person"]*len(people) + [o[1] for o in objects]
        height, width = frame.shape[:2]
        b = boxes.copy()
        b[:, [0, 2]] /= width
        b[:, [1, 3]] /= height
        normalized = np.column_stack(((b[:, 0]+b[:, 2])/2, (b[:, 1]+b[:, 3])/2,
                                       b[:, 2]-b[:, 0], b[:, 3]-b[:, 1]))
        padded = np.zeros((1, self.meta["max_boxes"], 4), np.float32)
        padded[0, :len(boxes)] = normalized
        size = self.meta["img_size"]
        image = cv2.resize(frame, (size, size))[:, :, ::-1].transpose(2, 0, 1)[None]
        feed = {"image": np.ascontiguousarray(image, np.float32)/255, "boxes": padded,
                "box_counts": np.array([len(boxes)], np.int64), "W": self.W, "alpha": self.alpha}
        results = dict(zip([item.name for item in self.relation.get_outputs()], self.relation.run(None, feed)))
        pred, pair = results["pred_logits"][0], results["pair_logits"][0]
        sub, obj, valid = results["sub_idx"][0], results["obj_idx"][0], results["valid_mask"][0]
        cal = self.meta["calibration"]
        scores = 1/(1+np.exp(-np.clip(cal["a"]*(pred+pair[:, None])+cal["b"], -60, 60)))
        for k in range(len(valid)):
            si, oi = int(sub[k]), int(obj[k])
            if not valid[k] or not 0 <= si < len(people) or not len(people) <= oi < len(boxes):
                continue
            p = int(scores[k].argmax())
            score = float(scores[k, p])
            identity = people[si][0]
            if score < self.config.confidence or score <= (output[identity].confidence or 0):
                continue
            predicate, label = self.predicates[p], labels[oi]
            phrase = f"{predicate.capitalize()} {label}"
            output[identity] = SceneEvidence(
                status="matched", label=phrase, predicate=predicate, object_label=label, confidence=score,
                sampled_at=timestamp, age_seconds=0,
            )
        elapsed = (perf_counter()-start)*1000
        return {identity: evidence.model_copy(update={"latency_ms": elapsed})
                for identity, evidence in output.items()}


@dataclass
class SceneJob:
    frame: np.ndarray
    people: list[tuple[int, list[float]]]
    timestamp: float
    session: str


class SceneWorker:
    def __init__(self, backend: SceneBackend):
        self.backend = backend
        self.condition = threading.Condition()
        self.pending: SceneJob | None = None
        self.results: dict[int, SceneEvidence] = {}
        self.session: str | None = None
        self.last_submit = -float("inf")
        self.closed = False
        self.error: str | None = None
        self.thread = threading.Thread(target=self._run, name="sidp-relations", daemon=True)
        self.thread.start()

    def submit(self, frame, people, timestamp, session):
        with self.condition:
            if self.closed or timestamp-self.last_submit < self.backend.config.interval_seconds:
                return
            self.pending = SceneJob(frame.copy(), [(t.id, t.pose.bbox.as_list()) for t in people], timestamp, session)
            self.last_submit = timestamp
            self.condition.notify()

    def snapshot(self, identity: int, timestamp: float, session: str, min_timestamp: float | None = None):
        with self.condition:
            result = self.results.get(identity) if self.session == session else None
            error = self.error
        if error:
            return SceneEvidence(status="unavailable")
        # A late background result must not cross a person's continuity reset.
        if result is None or (min_timestamp is not None and (
            result.sampled_at is None or result.sampled_at < min_timestamp
        )):
            return SceneEvidence()
        age = max(0, timestamp - (result.sampled_at if result.sampled_at is not None else timestamp))
        if age > self.backend.config.max_age_seconds:
            return SceneEvidence(status="stale", age_seconds=age)
        return result.model_copy(update={"age_seconds": age})

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending is not None)
                if self.closed:
                    return
                job, self.pending = self.pending, None
            try:
                results = self.backend.infer(job.frame, job.people, job.timestamp)
                with self.condition:
                    self.results, self.session, self.error = results, job.session, None
            except Exception as exc:
                logger.exception("scene_inference_failed")
                with self.condition:
                    self.results, self.error = {}, str(exc)

    def close(self):
        with self.condition:
            self.closed, self.pending = True, None
            self.condition.notify_all()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("Scene inference worker did not stop within 5 seconds")
