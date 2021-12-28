import json
import os
import ast
from Modules.dummy.example import test

class Dummy:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        # TODO
        #   - initialize and load model here
        # model_path = os.path.join(self.path, "model.txt")
        # self.model = open(model_path, "r")
        pass

    def inference_by_data(self, module_results):
        module_results = ast.literal_eval(module_results)
        result = []
        # TODO
        #   - Inference using module results
        #   - how to use module_results
        #     for data in module_results :
        #         aggregation_result[data] # Using this data
        result = {"aggregation_result": [
            {
                # 1 timestamp & multiple class
                'label': [
                    {'description': 'word_name', 'score': 1.0},
                    {'description': 'word_name', 'score': 1.0}
                ],
            },
            {
                # 1 timestamp & 1 class
                'label': [
                    {'description': 'word_name', 'score': 1.0}
                ],
            }
        ]}
        self.result = result

        return self.result