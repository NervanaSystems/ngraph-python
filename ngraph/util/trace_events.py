# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import json
import os


def is_tracing_enabled():
    return ('TRACING' in os.environ and os.environ['TRACING'] == '1')


class TraceEventTracker(object):
    """Event tracing using Chrome's Trace Event format"""

    def __init__(self, tracker_name):
        """Construct a new Trace Event Tracker"""
        self.name = tracker_name
        self.events = []

    def add_event(self, ph, category, name, pid, tid, timestamp):
        """Add a Trace Event"""
        event = {}
        event['ph'] = ph
        event['cat'] = category
        event['name'] = name
        event['pid'] = pid
        event['tid'] = tid
        event['ts'] = timestamp
        return event

    def add_operation(self, category, name, pid, tid, timestamp, duration, args):
        """Add a computation operation event to the trace"""
        event = self.add_event('X', category, name, pid, tid, timestamp)
        event['dur'] = duration
        event['args'] = args
        self.events.append(event)

    def serialize(self):
        """Serialize the event trace to a string"""
        trace = {}
        trace['traceEvents'] = self.events
        return json.dumps(trace, separators=(',', ':'))

    def serialize_to_file(self):
        """Serialize the event trace to a file"""
        with open(self.name + ".json", 'w') as out:
            out.write(self.serialize())
