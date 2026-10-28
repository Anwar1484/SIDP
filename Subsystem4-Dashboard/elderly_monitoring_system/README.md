# [Start Bootstrap - SB Admin 2](https://startbootstrap.com/theme/sb-admin-2/)

[SB Admin 2](https://startbootstrap.com/theme/sb-admin-2/) is an open source admin dashboard theme for [Bootstrap](https://getbootstrap.com/) created by [Start Bootstrap](https://startbootstrap.com/).

For the legacy Bootstrap 3 version of this theme, you can view the [last stable release](https://github.com/StartBootstrap/startbootstrap-sb-admin-2/releases/tag/v3.3.7%2B1) of SB Admin 2 for Bootstrap 3.

## Preview

[![SB Admin 2 Preview](https://assets.startbootstrap.com/img/screenshots/themes/sb-admin-2.png)](https://startbootstrap.github.io/startbootstrap-sb-admin-2/)

**[Launch Live Preview](https://startbootstrap.github.io/startbootstrap-sb-admin-2/)**

## Status

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/StartBootstrap/startbootstrap-sb-admin-2/master/LICENSE)
[![npm version](https://img.shields.io/npm/v/startbootstrap-sb-admin-2.svg)](https://www.npmjs.com/package/startbootstrap-sb-admin-2)
[![Build Status](https://travis-ci.org/StartBootstrap/startbootstrap-sb-admin-2.svg?branch=master)](https://travis-ci.org/StartBootstrap/startbootstrap-sb-admin-2)
[![dependencies Status](https://david-dm.org/StartBootstrap/startbootstrap-sb-admin-2/status.svg)](https://david-dm.org/StartBootstrap/startbootstrap-sb-admin-2)
[![devDependencies Status](https://david-dm.org/StartBootstrap/startbootstrap-sb-admin-2/dev-status.svg)](https://david-dm.org/StartBootstrap/startbootstrap-sb-admin-2?type=dev)

## Download and Installation

To begin using this template, choose one of the following options to get started:

* [Download the latest release on Start Bootstrap](https://startbootstrap.com/theme/sb-admin-2/)
* Install via npm: `npm i startbootstrap-sb-admin-2`
* Clone the repo: `git clone https://github.com/StartBootstrap/startbootstrap-sb-admin-2.git`
* [Fork, Clone, or Download on GitHub](https://github.com/StartBootstrap/startbootstrap-sb-admin-2)

## Usage

After installation, run `npm install` and then run `npm start` which will open up a preview of the template in your default browser, watch for changes to core template files, and live reload the browser when changes are saved. You can view the `gulpfile.js` to see which tasks are included with the dev environment.

## Elderly posture monitoring dashboard

The `dashboard.html` page uses Firebase Authentication for staff sign-in, Firebase Realtime Database for resident profiles and monitoring events, and Firebase Storage for optional resident profile photos.

### Firebase setup

1. In Firebase Console, enable **Authentication → Email/Password** and create staff accounts. Public account registration is intentionally not enabled.
2. Create a **Realtime Database** and copy its database URL into `databaseURL` in `js/firebase-config.js`. Keep the existing project's other Firebase config values.
3. In Realtime Database **Rules**, publish the contents of `database.rules.json`. These rules require a signed-in user to read or write data; only create accounts for authorized staff.
4. In Firebase Console, open **Storage → Rules** and publish `storage.rules`. Enable/create the Storage bucket if the project has not used Firebase Storage before. The rules allow signed-in staff to read warning snapshots; snapshot uploads are JPEGs limited to 1 MB each.
5. Serve this folder through a local web server (for example, VS Code Live Server), then open `login.html`. ES modules and Firebase do not work reliably from a `file://` URL.

### Local development and HTTPS hosting

- **Run locally:** install the VS Code Live Server extension, right-click `login.html`, and choose **Open with Live Server**. Open `http://127.0.0.1:5500/login.html` (the port may differ). `localhost` and `127.0.0.1` are treated as secure contexts by modern browsers, so Firebase modules and sign-in can be tested over local HTTP. Add `localhost` to Firebase Authentication's authorized domains if it is missing.
- **Publish an HTTPS site:** Firebase Hosting is configured in `firebase.json` for this project. Install the Firebase CLI with `npm install --global firebase-tools`, then run `firebase login` and `firebase deploy --only hosting,database,storage` from the project folder. Firebase prints the HTTPS `web.app` URL after deployment and provisions HTTPS automatically. Add the hosted domain in **Authentication → Settings → Authorized domains** if Firebase requests it.
- **Keep access protected:** the website files are public on Hosting, but resident records remain restricted by Firebase Authentication and the database/storage rules. Only create accounts for authorized staff. Do not upload real resident information to an untrusted public demo or development project.

To preview without signing in or touching Firebase, open `dashboard.html?demo=1` through Live Server, or choose **Explore the sample dashboard** on the login page. This mode uses temporary synthetic data in the browser only.

### Realtime Database data shape

- `RESIDENTS/{residentId}`: `full_name`, `age`, `address`, `room_number`, `risk_category`, optional `photo_storage_path`, and `created_at`. The dashboard uploads optional JPG/PNG/WebP photos (maximum 5 MB) to `RESIDENT_PHOTOS/{residentId}/` in Firebase Storage and writes the storage path here. Existing/downloadable demo profiles may also have `photo_url`. An SVG placeholder is used when no photo is saved.
- `CURRENT_POSTURE_STATUS/{residentId}`: `posture` or `pose_status`, `activity_recognition` (sit, stand, walk, bend, or lying), `angle_degrees`, `updated_at`, and optional `fall_detected` / `inactivity_detected` booleans. The dashboard subscribes to this node for live resident status. Use the Firebase resident key for `{residentId}`.
- `ACTIVITY_LOGS/{logId}`: `resident_id`, `timestamp` (ISO date-time or milliseconds since epoch), `event_type`, `activity_recognition`, `pose_status`, `angle_degrees`, `fall_detected`, `inactivity_detected`, `alert_type`, `severity`, and optional `notes`. Write warning, fall, inactivity, and unsafe-posture events here only; do not persist routine activity samples. The dashboard queries at most the latest 100 records per resident, filters out non-warning events, and applies the inclusive local date range.
- Warning snapshots are uploaded only when a caretaker selects **Capture now** on a warning while the browser camera is live. Images are resized to at most 1280×720, JPEG-compressed, and stored at `POSTURE_ALERT_SNAPSHOTS/{residentId}/{logId}.jpg`. The path and capture time are attached to the corresponding activity log, where authorized caretakers can open the snapshot. Demo snapshots remain in that browser session and are not uploaded.

The date-filtered summary and history contain warning and unsafe-posture events only. The Excel export downloads an `.xlsx` workbook with Summary and Warnings sheets for the selected resident and applied date range. The export uses SheetJS from jsDelivr, so an internet connection is required.

### Synthetic test data

`database.demo.json` contains clearly labelled, fictional sample profiles and monitoring events for importing into a **separate, empty test Realtime Database only**. In Firebase Console, open that test database's Data view and use **Import JSON** at the root. Importing at the root replaces that database's current data; do not import this file into a database containing real residents. For the no-write browser preview, use `dashboard.html?demo=1` instead.

Coordinate the field names and resident IDs with Subsystem 3 before connecting its sensor publisher. The Original panel requests the current browser device camera and displays it unmasked; the video feed is not uploaded. The adjacent skeleton panel is a placeholder until an AI keypoint stream is connected. For a remote room camera, connect a browser-compatible stream through an approved camera-streaming service rather than storing video frames in Realtime Database.

Firebase web configuration is visible to browser users by design; Realtime Database security rules and controlled staff accounts protect the data. Do not use open read/write rules for resident or posture data.

### Gulp Tasks

* `gulp` the default task that builds everything
* `gulp watch` browserSync opens the project in your default browser and live reloads when changes are made
* `gulp css` compiles SCSS files into CSS and minifies the compiled CSS
* `gulp js` minifies the themes JS file
* `gulp vendor` copies dependencies from node_modules to the vendor directory

You must have npm installed globally in order to use this build environment. This theme was built using node v11.6.0 and the Gulp CLI v2.0.1. If Gulp is not running properly after running `npm install`, you may need to update node and/or the Gulp CLI locally.

## Bugs and Issues

Have a bug or an issue with this template? [Open a new issue](https://github.com/StartBootstrap/startbootstrap-sb-admin-2/issues) here on GitHub or leave a comment on the [template overview page at Start Bootstrap](https://startbootstrap.com/theme/sb-admin-2/).

## About

Start Bootstrap is an open source library of free Bootstrap templates and themes. All of the free templates and themes on Start Bootstrap are released under the MIT license, which means you can use them for any purpose, even for commercial projects.

* <https://startbootstrap.com>
* <https://twitter.com/SBootstrap>

Start Bootstrap was created by and is maintained by **[David Miller](https://davidmiller.io/)**.

* <https://davidmiller.io>
* <https://twitter.com/davidmillerhere>
* <https://github.com/davidtmiller>

Start Bootstrap is based on the [Bootstrap](https://getbootstrap.com/) framework created by [Mark Otto](https://twitter.com/mdo) and [Jacob Thorton](https://twitter.com/fat).

## Copyright and License

Copyright 2013-2021 Start Bootstrap LLC. Code released under the [MIT](https://github.com/StartBootstrap/startbootstrap-resume/blob/master/LICENSE) license.
