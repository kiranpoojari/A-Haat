# views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
import os
import uuid

from .ml.leaf_engine import predict_images as leaf_predict
from .ml.areca_coconut_engine import predict_images as areca_predict


class LeafHealthAPIView(APIView):
    """
    POST:
    - crop
    - images[]
    """

    def post(self, request):
        crop = request.data.get("crop")
        images = request.FILES.getlist("images")

        if not crop or len(images) < 3:
            return Response(
                {"error": "Crop and minimum 3 images required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        temp_paths = []

        try:
            for img in images:
                filename = f"{uuid.uuid4()}_{img.name}"
                path = os.path.join(settings.MEDIA_ROOT, filename)

                with open(path, "wb+") as f:
                    for chunk in img.chunks():
                        f.write(chunk)

                temp_paths.append(path)

            result = leaf_predict(temp_paths, crop.capitalize())
            return Response(result, status=status.HTTP_200_OK)

        finally:
            for path in temp_paths:
                if os.path.exists(path):
                    os.remove(path)


class ArecaCoconutAPIView(APIView):
    """
    POST:
    - images[]
    """

    def post(self, request):
        images = request.FILES.getlist("images")

        if len(images) < 3:
            return Response(
                {"error": "Minimum 3 images required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        temp_paths = []

        try:
            for img in images:
                filename = f"{uuid.uuid4()}_{img.name}"
                path = os.path.join(settings.MEDIA_ROOT, filename)

                with open(path, "wb+") as f:
                    for chunk in img.chunks():
                        f.write(chunk)

                temp_paths.append(path)

            result = areca_predict(temp_paths)
            return Response(result, status=status.HTTP_200_OK)

        finally:
            for path in temp_paths:
                if os.path.exists(path):
                    os.remove(path)
