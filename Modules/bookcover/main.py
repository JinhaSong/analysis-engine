import os

from AnalysisEngine import settings
from Modules.dummy.example import test
from WebAnalyzer.utils.media import frames_to_timecode
from PIL import Image

class BookCover:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))
    class_names = ['19금', '라이프스타일', '부모', '에세이', '인문', '종교', '판타지무협', '경제경영', '매거진', '여행', '과학', '로맨스BL', '사회', '소설', '어린이청소년', '역사', '자기계발']

    def __init__(self):
        # TODO
        #   - initialize and load model here
        # model_path = os.path.join(self.path, "model.txt")
        # self.model = open(model_path, "r")
        model = EfficientNet.from_pretrained(model_name, num_classes=dataset_info['nc'])
        model.load_state_dict(torch.load(model_path))

        transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])

    def inference_by_image(self, image_path):
        result = []
        image = transform(Image.open(image_path)).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            outputs = model(image)

        for idx in torch.topk(outputs, k=topk).indices.squeeze(0).tolist():
            prob = torch.softmax(outputs, dim=1)[0, idx].item()
            result.append({'class': self.class_names[idx], 'prob': prob, "class_idx": idx})
        self.result = result

        return self.result