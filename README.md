# Mi API de clasificación de imágenes

Modelo CNN (CIFAR-10) cuantizado a int8 y servido con FastAPI + ONNX Runtime.

- `/health`: comprueba que el servicio está vivo
- `/docs`: página de pruebas
- `/predict`: recibe una imagen y devuelve la clase
