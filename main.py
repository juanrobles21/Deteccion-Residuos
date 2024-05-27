# Librerias
from tkinter import *
from PIL import Image, ImageTk
import imutils
import cv2
import numpy as np
import math
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torch.autograd import Variable

# Función para cargar el modelo ResNet
def cargar_modelo(ruta_archivo):
    modelo = ResNet()
    try:
        checkpoint = torch.load(ruta_archivo, map_location=torch.device('cpu'))
        modelo.load_state_dict(checkpoint['model_state_dict'])
        print("Modelo cargado exitosamente.")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
    return modelo

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.network = resnet50(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 5)

    def forward(self, x):
        return self.network(x)

# Scanning Function
def Scanning():
    global lblVideo
    if cap is not None:
        ret, frame = cap.read()
        frame_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret == True:
            clase = detectar_residuos(frame_show, modelo)

            mostrar_deteccion(frame_show, clase)
            # Resize
            frame_show = imutils.resize(frame_show, width=640)
            # Convertir Video
            im = Image.fromarray(frame_show)
            img = ImageTk.PhotoImage(image=im)

            # Mostrar img
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, Scanning)
        else:
            cap.release()
#
def mostrar_deteccion(frame, clase):
    # Obtener las dimensiones del frame
    height, width, _ = frame.shape

    # Definir colores para cada clase (puedes ajustar estos colores según tu preferencia)
    color = [(231, 9, 58),(36, 171, 56),(249, 233, 0), (52, 146, 209), (126, 117, 128)]

    # Dibujar un cuadro alrededor del objeto detectado
    cv2.rectangle(frame, (0, 0), (width, height), color[clase], 10)

    # Mostrar la detección en la interfaz gráfica
    im = Image.fromarray(frame)
    img = ImageTk.PhotoImage(image=im)
    lblVideo.configure(image=img)
    lblVideo.image = img


# Función para preprocesar la imagen antes de pasarla al modelo
def preprocesar_imagen(img):
    transformacion = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img = transformacion(img)
    img = Variable(img.unsqueeze(0))
    return img

# Función para realizar la detección de residuos con ResNet
def detectar_residuos(frame, modelo):
    global lblimg, lblimginfo
    # Interfaz
    lblimg = Label(pantalla)
    lblimg.place(x=27, y=273)

    lblimginfo = Label(pantalla)
    lblimginfo.place(x=1057, y=335)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(frame)
    img = preprocesar_imagen(im)
    modelo.eval()
    with torch.no_grad():
        outputs = modelo(img)
    _, preds = torch.max(outputs, dim=1)
    clase = int(preds.item())
    if clase == 1:
        images(img_vidrio,img_vidriotxt)
    elif clase == 2:
        images(img_metal,img_metaltxt)
    elif clase == 3:
        images(img_plastico,img_plasticotxt)
    elif clase == 4:
        images(img_carton,img_cartontxt)
    else:
        images(img_no_encontrado,img_no_encontradotxt)
    return clase


# Show Images
def images(img,imginfo):
    img = img

    # Img detect
    img = np.array(img, dtype="uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)

    img_ = ImageTk.PhotoImage(image=img)
    lblimg.configure(image=img_)
    lblimg.image = img_

    imginfo = imginfo

    # Img detect
    imginfo = np.array(imginfo, dtype="uint8")
    imginfo = cv2.cvtColor(imginfo, cv2.COLOR_RGB2BGR)
    imginfo = Image.fromarray(imginfo)

    imginfo_ = ImageTk.PhotoImage(image=imginfo)
    lblimginfo.configure(image=imginfo_)
    lblimginfo.image = imginfo_


# main
def ventana_principal():
    global modelo, lblVideo, cap, pantalla
    global img_vidrio,img_metal,img_plastico,img_carton, img_no_encontrado
    global img_vidriotxt,img_metaltxt, img_plasticotxt, img_cartontxt, img_no_encontradotxt
    # Ventana principal
    pantalla = Tk()
    pantalla.title("DETECCIÓN DE RESIDUOS")
    pantalla.geometry("1280x720")

    # Background
    img_background = PhotoImage(file="setUp/Background.png")
    background = Label(image=img_background)
    background.place(x=0, y=0, relwidth=1, relheight=1)

    # Model
    ruta_archivo = 'Models/model_checkpointfinal.pt'
    modelo = cargar_modelo(ruta_archivo)

    # Img

    img_vidrio = cv2.imread("setUp/PapeleraVidrio.png")
    img_metal = cv2.imread("setUp/PapelaraMetal.png")
    img_plastico = cv2.imread("setUp/PapeleraPlastico.png")
    img_carton = cv2.imread("setUp/PapeleraCarton.png")
    img_no_encontrado = cv2.imread("setUp/ImgNoEncontrado.png")

    img_vidriotxt = cv2.imread("setUp/InfoVidrio.png")
    img_metaltxt = cv2.imread("setUp/InfoMetal.png")
    img_plasticotxt = cv2.imread("setUp/InfoPapePla.png")
    img_cartontxt = cv2.imread("setUp/InfoCarton.png")
    img_no_encontradotxt = cv2.imread("setUp/InfoNoEncontrado.png")

    # Label Video
    lblVideo = Label(pantalla)
    lblVideo.place(x=320, y=165)

    # Cam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 640)
    cap.set(4, 480)

    # Scanning
    Scanning()

    # Loop
    pantalla.mainloop()

if __name__ == "__main__":
    ventana_principal()
