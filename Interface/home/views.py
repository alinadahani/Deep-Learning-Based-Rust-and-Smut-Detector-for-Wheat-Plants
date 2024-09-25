from django.shortcuts import render
from .image import handle_image_upload
from .video import process_video
from .directory import count_images_in_directory
import os

def image(request):
    return render(request, 'image.html', {'current_page': 'image'})

def video(request):
    return render(request, 'video.html', {'current_page': 'video'})

def directory(request):
    return render(request, 'directory.html', {'current_page': 'directory'})

def memories(request):
    return render(request, 'memories.html', {'current_page': 'memories'})

def predict_image(request):
    if request.method == 'POST':
        image_file = request.FILES.get('media')
        if image_file:
            image_base64, message = handle_image_upload(image_file)
            if image_base64:
                return render(request, 'image.html', {
                    'image_base64': image_base64,
                    'predicted_label': message,
                    'current_page': 'image'
                })
            else:
                return render(request, 'image.html', {
                    'error': message,
                    'current_page': 'image'
                })
    return render(request, 'image.html', {'current_page': 'image'})

def predict_video(request):
    if request.method == 'POST' and request.FILES['media'] and 'option' in request.POST:
        video_file = request.FILES['media']
        selected_option = request.POST['option']
        frame_count, label_counts, error = process_video(video_file, selected_option)

        if error:
            return render(request, 'video.html', {'error': error, 'current_page': 'video'})

        return render(request, 'video.html', {
            'frame_count': frame_count,
            'label_counts': label_counts,
            'selected_option': selected_option,
            'current_page': 'video'
        })

    return render(request, 'video.html', {'current_page': 'video'})



def predict_directory(request):
    if request.method == 'POST' and 'directory_path' in request.POST and 'option' in request.POST:
        directory_path = request.POST['directory_path']
        selected_option = request.POST['option']

        if not os.path.isdir(directory_path):
            return render(request, 'directory.html', {'error': 'Invalid directory path', 'current_page': 'directory'})

        image_count, label_counts, error = count_images_in_directory(directory_path, selected_option)
        
        if error:
            return render(request, 'directory.html', {'error': error, 'current_page': 'directory'})

        return render(request, 'directory.html', {
            'image_count': image_count,
            'label_counts': label_counts,
            'selected_option': selected_option,
            'current_page': 'directory'
        })

    return render(request, 'directory.html', {'current_page': 'directory'})
