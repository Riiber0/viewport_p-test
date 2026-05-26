from collections import defaultdict
from utils import get_viewport_tiles_rad
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom

GRAPH_PATH = "./graphs/"

class NavigationGraphPredictor:

    def __init__(self, pw=5,graph_file=None):
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.transition_probs = {}
        self.pw = pw
        self.state_to_id = {}
        self.id_to_state = {}

        if graph_file != None:
            self.import_xml(graph_file)

    def viewport_to_state(self, tiles):
        return tuple(sorted(tiles))

    def import_xml(self, xml_file):

        self.state_to_id = {}
        self.id_to_state = {}

        tree = ET.parse(xml_file)

        root = tree.getroot()

        states_xml = root.find("States")

        for state_node in states_xml.findall("State"):

            state_id = int(state_node.get("id"))

            tiles = tuple(
                map(
                    int,
                    state_node.get("tiles").split(",")
                )
            )

            self.state_to_id[tiles] = state_id
            self.id_to_state[state_id] = tiles

        transitions_xml = root.find("Transitions")

        for trans_node in transitions_xml.findall("Transition"):

            src_id = int(trans_node.get("from"))
            dst_id = int(trans_node.get("to"))

            prob = float(
                trans_node.get("probability")
            )

            src_state = self.id_to_state[src_id]
            dst_state = self.id_to_state[dst_id]

            self.transition_probs.setdefault(
                src_state,
                {}
            )

            self.transition_probs[src_state][dst_state] = prob

        #print(f"{xml_file} import finished")

    def export_xml(self, output_file="navigation_graph.xml"):

        root = ET.Element("NavigationGraph")

        states_xml = ET.SubElement(root, "States")

        for state, state_id in self.state_to_id.items():

            state_node = ET.SubElement(states_xml, "State")

            state_node.set("id", str(state_id))
            state_node.set(
                "tiles",
                ",".join(map(str, state))
            )

        transitions_xml = ET.SubElement(root, "Transitions")

        for s1, next_states in self.transition_probs.items():

            src_id = self.state_to_id[s1]

            for s2, prob in next_states.items():

                dst_id = self.state_to_id[s2]

                trans_node = ET.SubElement(
                    transitions_xml,
                    "Transition"
                )

                trans_node.set("from", str(src_id))
                trans_node.set("to", str(dst_id))
                trans_node.set("probability", str(prob))

        xml_str = ET.tostring(root, encoding="utf-8")

        pretty_xml = minidom.parseString(xml_str).toprettyxml(
            indent="    "
        )

        with open(output_file, "w") as f:
            f.write(pretty_xml)

        #print(f"XML salvo em: {output_file}")

    def fit(self, viewport_sequence):

        states = [
            self.viewport_to_state(vp)
            for vp in viewport_sequence
        ]

        for state in states:
            if state not in self.state_to_id:

                idx = len(self.state_to_id)

                self.state_to_id[state] = idx
                self.id_to_state[idx] = state

        for s1, s2 in zip(states[:-1], states[1:]):
            self.transition_counts[s1][s2] += 1

        for s1, next_states in self.transition_counts.items():

            total = sum(next_states.values())

            self.transition_probs[s1] = {}

            for s2, count in next_states.items():
                self.transition_probs[s1][s2] = count / total

    def predict_state_probs(self, current_viewport):

        state = self.viewport_to_state(current_viewport)

        if state not in self.transition_probs:
            #TODO
            return {}

        return self.transition_probs[state]

    def predict_tile_probs(self, current_viewport):

        state_probs = self.predict_state_probs(current_viewport)

        tile_probs = defaultdict(float)

        for next_state, prob in state_probs.items():

            for tile in next_state:
                tile_probs[tile] += prob

        total = sum(tile_probs.values())

        if total > 0:
            for tile in tile_probs:
                tile_probs[tile] /= total

        return dict(tile_probs)

    def predict(self, current_viewport):

        #for i in range(0, int(self.pw*10)):
        state_probs = self.predict_state_probs(current_viewport)
        tiles_probs = self.predict_tile_probs(current_viewport)
        current_viewport = list(tiles_probs.keys())

        return list(current_viewport)

if __name__ == '__main__':
    train_df = pd.read_csv('train_data.csv')
    val_df = pd.read_csv('val_data.csv')
    test_df = pd.read_csv('test_data.csv')
    pws = [5, 10, 15,20 , 25]

    for pw in pws:
        for v_id, video_df in train_df.groupby('v_id'):
            for u_id, session in video_df.groupby('u_id'):
                traces = list()
                for start in range(0, pw):
                    for _, row in session[start::pw].iterrows():

                        pitch = float(row['pitch'])
                        yaw = float(row['yaw'])

                        viewport_tiles = get_viewport_tiles_rad(
                            pitch_rad=pitch,
                            yaw_rad=yaw
                        )

                        traces.append(viewport_tiles)

                    predictor = NavigationGraphPredictor()
                    predictor.fit(traces)

            predictor.export_xml(GRAPH_PATH+f'video_{v_id}_{pw}_ng.xml')

    for pw in pws:
        total = 0
        miss = 0
        for v_id, traces in val_df.groupby('v_id'):
            predictor = NavigationGraphPredictor(pw, GRAPH_PATH+f'video_{v_id}_{pw}_ng.xml')
            
            for start in range(0, pw):
                for _, row in traces[start::pw].iterrows():
                    pred = None
                    pitch = float(row['pitch'])
                    yaw = float(row['yaw'])

                    viewport_tiles = get_viewport_tiles_rad(
                        pitch_rad=pitch,
                        yaw_rad=yaw
                    )

                    pred = predictor.predict(viewport_tiles)

                    if pred is not None:
                        miss += len(set(pred) - set(viewport_tiles))
                        #miss += len(set(viewport_tiles) - set(pred))
                        total += len(viewport_tiles)

                    pred = predictor.predict(viewport_tiles)

        accuracy = 1 - (miss / total)
        print(f'pw = {pw} | validation accuracy: {accuracy}')

    for pw in pws:
        total = 0
        miss = 0
        for v_id, traces in test_df.groupby('v_id'):
            predictor = NavigationGraphPredictor(pw, GRAPH_PATH+f'video_{v_id}_{pw}_ng.xml')

            for start in range(0, pw):
                for _, row in traces[start::pw].iterrows():
                    pred = None
                    pitch = float(row['pitch'])
                    yaw = float(row['yaw'])

                    viewport_tiles = get_viewport_tiles_rad(
                        pitch_rad=pitch,
                        yaw_rad=yaw
                    )

                    pred = predictor.predict(viewport_tiles)

                    if pred is not None:
                        miss += len(set(pred) - set(viewport_tiles))
                        #miss += len(set(viewport_tiles) - set(pred))
                        total += len(viewport_tiles)

                    pred = predictor.predict(viewport_tiles)

        accuracy = 1 - (miss / total)
        print(f'pw = {pw} | validation accuracy: {accuracy}')

