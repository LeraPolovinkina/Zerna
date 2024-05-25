from django.http import JsonResponse, HttpResponse
from django.conf import settings
import os
import torch
from django.shortcuts import render
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(settings.BASE_DIR, 'model_weights_epoch_10.pth')
class_names = ['barley', 'flax', 'oats', 'wheat']

def upload_image(request):
    if request.method == 'GET':
        return render(request, 'index.html')
    elif request.method == 'POST':
        return HttpResponse("Image uploaded successfully!")

def predict_one_sample(model, input_tensor):
    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(DEVICE)
        outputs = model(input_tensor).cpu()
    probs = torch.nn.functional.softmax(outputs, dim=-1).numpy()
    return probs

def classify_image(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        image_path = os.path.join(settings.MEDIA_ROOT, image.name)
        with open(image_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0)
        resnet = resnet50().to(DEVICE)
        resnet.fc = torch.nn.Linear(resnet.fc.in_features, 45)
        resnet.load_state_dict(torch.load(model_path, map_location=DEVICE))
        prob_pred = predict_one_sample(resnet, image_tensor)
        predicted_proba = np.max(prob_pred) * 100
        y_pred = np.argmax(prob_pred)
        predicted_label = class_names[y_pred]
        return JsonResponse({'image_url': image_path, 'classification': predicted_label, 'accuracy': predicted_proba})
