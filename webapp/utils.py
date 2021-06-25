from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import json


def singleimg_embedding(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (240, 240))

    # Initializing for face detection
    mtcnn = MTCNN(image_size=240, margin=0, device='cuda', keep_all=False)

    resnetv1 = InceptionResnetV1(pretrained='vggface2').eval()
    face = mtcnn(img)
    emb = resnetv1(
        face.unsqueeze(0))
    emb = emb.detach().cpu().numpy()
    emb = emb.tolist()
    emb = json.dumps(emb)

    return emb