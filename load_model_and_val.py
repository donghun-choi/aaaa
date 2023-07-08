from tensorflow.keras.models import load_model
import cv2
# Load the model
loaded_model = load_model('abalone_model.h5')

def load_img(index):
    # 이미지 경로 설정
    image_path = './frames/{}.jpg'.format(index)
    
    # 이미지 불러오기
    img = cv2.imread(image_path)
    
    # 이미지 크기 조정
    img = cv2.resize(img, (224, 224))
    
    # Canny Edge Detection 적용
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)
    
    return edges

# Now you can use the loaded model to predict the WASD values for a new image
def predict_wasd(edges):
    edges = edges.reshape(-1, 224, 224, 1)
    
    # Use the model to predict the WASD values
    wasd_values = loaded_model.predict(edges)
    
    return wasd_values


print(predict_wasd(load_img(1222)))