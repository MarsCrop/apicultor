[
    {
        "id": "",
        "label": "MIR State",
        "widgets": [
            {
                "type": "fader",
                "id": "volumen",
                "linkId": "",
                "label": "auto",
                "unit": "",
                "left": 0,
                "top": 0,
                "width": 100,
                "height": 350,
                "alignRight": false,
                "horizontal": false,
                "noPip": false,
                "compact": false,
                "color": "auto",
                "css": "",
                "snap": false,
                "spring": false,
                "range": {
                    "min": 0,
                    "max": 3
                },
                "origin": "auto",
                "value": "",
                "logScale": true,
                "precision": 2,
                "meter": true,
                "address": "/volume",
                "preArgs": [],
                "target": []
            },
            {
                "type": "fader",
                "id": "High Frequency Content",
                "linkId": "",
                "label": "auto",
                "unit": "",
                "left": 210,
                "top": 0,
                "width": 530,
                "height": 90,
                "alignRight": false,
                "horizontal": true,
                "noPip": false,
                "compact": false,
                "color": "auto",
                "css": "",
                "snap": false,
                "spring": false,
                "range": {
                    "min": 0,
                    "max": 1
                },
                "origin": "auto",
                "value": "",
                "logScale": false,
                "precision": 2,
                "meter": false,
                "address": "/hfc",
                "preArgs": [],
                "target": []
            },
            {
                "type": "push",
                "id": "New sample",
                "label": "auto",
                "left": 750,
                "top": 0,
                "width": "auto",
                "height": 350,
                "color": "auto",
                "css": "",
                "precision": 2,
                "address": "/gate",
                "preArgs": [],
                "target": [],
                "on": 1,
                "off": 0,
                "linkId": "",
                "norelease": false
            },
            {
                "type": "knob",
                "id": "Num Harmonics",
                "linkId": "",
                "label": "auto",
                "unit": "",
                "left": 860,
                "top": 0,
                "width": 110,
                "height": 100,
                "noPip": false,
                "compact": false,
                "color": "auto",
                "css": "",
                "snap": false,
                "spring": false,
                "range": {
                    "min": 0,
                    "max": 50
                },
                "origin": "auto",
                "value": "",
                "logScale": false,
                "precision": 2,
                "address": "/nharm",
                "preArgs": [],
                "target": [],
                "angle": 270
            },
            {
                "type": "fader",
                "id": "Low Frequency Content",
                "linkId": "",
                "label": "auto",
                "unit": "",
                "left": 210,
                "top": 100,
                "width": 530,
                "height": 90,
                "alignRight": false,
                "horizontal": true,
                "noPip": false,
                "compact": false,
                "color": "auto",
                "css": "",
                "snap": false,
                "spring": false,
                "range": {
                    "min": 0,
                    "max": 1
                },
                "origin": "auto",
                "value": "",
                "logScale": false,
                "precision": 2,
                "meter": false,
                "address": "/fader_1",
                "preArgs": [],
                "target": []
            },
            {
                "type": "switch",
                "id": "LFC Mode",
                "linkId": "",
                "label": "auto",
                "left": 110,
                "top": 100,
                "width": "auto",
                "height": 90,
                "color": "auto",
                "css": "",
                "value": "",
                "precision": 2,
                "address": "/fader_2",
                "preArgs": [],
                "target": [],
                "horizontal": false,
                "values": {
                    ">": 1,
                    "<": 2
                }
            },
            {
                "type": "switch",
                "id": "HFC Mode",
                "label": "auto",
                "left": 110,
                "top": 0,
                "width": "auto",
                "height": 90,
                "color": "auto",
                "css": "",
                "value": "",
                "precision": 2,
                "address": "/fader_2",
                "preArgs": [],
                "target": [],
                "horizontal": false,
                "values": {
                    ">": 1,
                    "<": 2
                },
                "linkId": ""
            },
            {
                "type": "knob",
                "id": "Duration",
                "label": "auto",
                "unit": "",
                "left": 210,
                "top": 200,
                "width": 160,
                "height": 150,
                "noPip": false,
                "compact": false,
                "color": "auto",
                "css": "",
                "snap": false,
                "spring": false,
                "range": {
                    "min": 0,
                    "max": 1
                },
                "origin": "auto",
                "value": "",
                "logScale": false,
                "precision": 2,
                "address": "/fader_1",
                "preArgs": [],
                "target": [],
                "linkId": "",
                "angle": 270
            },
            {
                "type": "switch",
                "id": "Mode",
                "label": "auto",
                "left": 110,
                "top": 200,
                "width": "auto",
                "height": 150,
                "color": "auto",
                "css": "",
                "value": "",
                "precision": 2,
                "address": "/fader_2",
                "preArgs": [],
                "target": [],
                "horizontal": false,
                "values": {
                    ">": 1,
                    "<": 2
                },
                "linkId": ""
            },
            {
                "type": "keyboard",
                "id": "pitch",
                "label": "auto",
                "left": 390,
                "top": 200,
                "width": 350,
                "height": 150,
                "color": "auto",
                "css": "",
                "precision": 2,
                "address": "/fader_3",
                "preArgs": [],
                "target": [],
                "keys": 24,
                "start": 60,
                "traversing": true,
                "on": 1,
                "off": 0,
                "split": false
            },
            {
                "type": "push",
                "id": "A",
                "linkId": "",
                "label": "auto",
                "left": 110,
                "top": 370,
                "width": 170,
                "height": 180,
                "color": "auto",
                "css": "",
                "precision": 2,
                "address": "/fader_1",
                "preArgs": [],
                "target": [],
                "on": 1,
                "off": 0,
                "norelease": false
            },
            {
                "type": "toggle",
                "id": "rec",
                "linkId": "",
                "label": "auto",
                "left": 0,
                "top": 370,
                "width": "auto",
                "height": "auto",
                "color": "auto",
                "css": "",
                "value": "",
                "precision": 2,
                "address": "/fader_2",
                "preArgs": [],
                "target": [],
                "on": 1,
                "off": 0
            },
            {
                "type": "toggle",
                "id": "B",
                "linkId": "",
                "label": "auto",
                "left": 290,
                "top": 370,
                "width": 180,
                "height": 180,
                "color": "auto",
                "css": "",
                "value": "",
                "precision": 2,
                "address": "/fader_3",
                "preArgs": [],
                "target": [],
                "on": 1,
                "off": 0
            },
            {
                "type": "toggle",
                "id": "C",
                "linkId": "",
                "label": "auto",
                "left": 480,
                "top": 370,
                "width": 170,
                "height": 180,
                "color": "auto",
                "css": "",
                "value": "",
                "precision": 2,
                "address": "/fader_4",
                "preArgs": [],
                "target": [],
                "on": 1,
                "off": 0
            },
            {
                "type": "toggle",
                "id": "D",
                "linkId": "",
                "label": "auto",
                "left": 660,
                "top": 370,
                "width": 190,
                "height": 180,
                "color": "auto",
                "css": "",
                "value": "",
                "precision": 2,
                "address": "/fader_5",
                "preArgs": [],
                "target": [],
                "on": 1,
                "off": 0
            }
        ]
    }
]