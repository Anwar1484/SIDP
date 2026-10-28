const hoursAgo = (hours) => new Date(Date.now() - hours * 60 * 60 * 1000).toISOString();

export const demoData = {
    RESIDENTS: {
        "demo-resident-001": {
            full_name: "Amina Sample",
            age: 72,
            room_number: "D-01",
            address: "DEMO ONLY - Sample address, Kuala Lumpur",
            risk_category: "Moderate",
            photo_url: "img/resident-placeholder.svg",
            created_at: hoursAgo(24 * 20)
        },
        "demo-resident-002": {
            full_name: "Ravi Sample",
            age: 81,
            room_number: "D-02",
            address: "DEMO ONLY - Sample address, Ipoh",
            risk_category: "High",
            photo_url: "img/resident-placeholder.svg",
            created_at: hoursAgo(24 * 15)
        },
        "demo-resident-003": {
            full_name: "Mei Sample",
            age: 76,
            room_number: "D-03",
            address: "DEMO ONLY - Sample address, Kuantan",
            risk_category: "Low",
            photo_url: "img/resident-placeholder.svg",
            created_at: hoursAgo(24 * 9)
        }
    },
    CURRENT_POSTURE_STATUS: {
        "demo-resident-001": {
            posture: "Upright",
            activity_recognition: "sitting",
            angle_degrees: 8,
            fall_detected: false,
            inactivity_detected: false,
            updated_at: hoursAgo(0.05)
        },
        "demo-resident-002": {
            posture: "Unsafe posture",
            activity_recognition: "lying",
            angle_degrees: 38,
            fall_detected: true,
            inactivity_detected: false,
            alert_active: true,
            updated_at: hoursAgo(0.15)
        },
        "demo-resident-003": {
            posture: "Upright",
            activity_recognition: "walking",
            angle_degrees: 5,
            fall_detected: false,
            inactivity_detected: false,
            updated_at: hoursAgo(0.02)
        }
    },
    ACTIVITY_LOGS: {
        "demo-log-001": {
            resident_id: "demo-resident-001",
            timestamp: hoursAgo(1),
            event_type: "Activity recognized",
            activity_recognition: "sit",
            pose_status: "Upright",
            angle_degrees: 8,
            fall_detected: false,
            inactivity_detected: false
        },
        "demo-log-002": {
            resident_id: "demo-resident-001",
            timestamp: hoursAgo(3),
            event_type: "Activity recognized",
            activity_recognition: "stand",
            pose_status: "Upright",
            angle_degrees: 12,
            fall_detected: false,
            inactivity_detected: false
        },
        "demo-log-003": {
            resident_id: "demo-resident-001",
            timestamp: hoursAgo(5),
            event_type: "Activity recognized",
            activity_recognition: "walk",
            pose_status: "Upright",
            angle_degrees: 7,
            fall_detected: false,
            inactivity_detected: false
        },
        "demo-log-004": {
            resident_id: "demo-resident-002",
            timestamp: hoursAgo(2),
            event_type: "Posture alert",
            activity_recognition: "bend",
            pose_status: "Unsafe posture",
            angle_degrees: 38,
            alert_active: true,
            alert_type: "Unsafe bending angle",
            severity: "Medium",
            notes: "DEMO ONLY - Example posture alert."
        },
        "demo-log-005": {
            resident_id: "demo-resident-002",
            timestamp: hoursAgo(4),
            event_type: "Fall detected",
            activity_recognition: "lying",
            pose_status: "Possible fall",
            angle_degrees: 72,
            fall_detected: true,
            severity: "High",
            notes: "DEMO ONLY - Example fall event."
        },
        "demo-log-006": {
            resident_id: "demo-resident-003",
            timestamp: hoursAgo(6),
            event_type: "Activity recognized",
            activity_recognition: "lying",
            pose_status: "Resting",
            angle_degrees: 4,
            fall_detected: false,
            inactivity_detected: false
        },
        "demo-log-007": {
            resident_id: "demo-resident-003",
            timestamp: hoursAgo(8),
            event_type: "Inactivity detected",
            activity_recognition: "sit",
            pose_status: "Still",
            inactivity_detected: true,
            severity: "Low",
            notes: "DEMO ONLY - Example prolonged inactivity event."
        }
    }
};
