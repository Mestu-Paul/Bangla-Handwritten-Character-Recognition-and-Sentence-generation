import tkinter as tk
from tkinter import filedialog
import io
from PIL import Image, ImageDraw, ImageTk
import tensorflow as tf
import numpy as np
from keras.models import load_model

root = tk.Tk()
root.geometry('1300x700+10+10')
root.title("Bangla Handwritten Alphabet to Sentence")

color = {
    'header-bg':'#FB5022',
    'header-fg':'white',
    'footer-bg':'#F53805',
    'footer-fg':'black'
}
alph = ["অ","আ","ই","ঈ","উ","ঊ","ঋ",
    "এ","ঐ","ও","ঔ","ক","খ","গ","ঘ","ঙ",
    "চ","ছ","জ","ঝ","ঞ","ট","ঠ","ড","ঢ",
    "ণ","ত","থ","দ","ধ","ন","প","ফ","ব",
    "ভ","ম","য","র","ল","শ","ষ","স","হ",
    "ড়","ঢ়","য়", "ৎ", "ং" , "ঃ", "ঁ"]
sentence = [
'অজগরটি আসছে তেড়ে' ,# অ
'আমটি আমি খাবো পেড়ে' ,# আ
'ইলিশ ভাজা খেতে মজা' ,# ই
'ঈগল হলো পাখির রাজা' ,# ঈ
'উট চলছে মরুর দেশে' ,# উ
'ঊষা হাসে পূর্বাকাশে' ,# ঊ
'ষড় ঋতুর বাংলাদেশ' ,# ঋ
'একতারাটি বাজে বেশ' ,# এ
'ঐরাবতটি আসছে তেড়ে' ,# ঐ
'ওজন করো সঠিক করে' ,# ও
'ঔষধ খাবো অসুখ হলে' ,# ঔ
'কমলা খেলে প্রাণ জুড়ায়' ,# ক
'খরগোশেরা লাফিয়ে বেড়ায়' ,# খ
'গোলাপ ফুল গন্ধ ছড়ায়' ,# গ
'ঘুঘু ডাকা নিরালায়' ,# ঘ
'ব্যাঙ থাকে ডোবার জলে' ,# ঙ
'চিতাবাঘ দ্রুত চলে' ,# চ
'ছাগল পোষা বেশ ঝামেলা' ,# ছ
'জিরাফের লম্বা গলা' ,# জ
'ঝিনুকের পেটে মুক্তা হয়' ,# ঝ
'মিঞার হাতে লাঠি রয়' ,# ঞ
'টিয়ার ঠোঁট বাকা হয়' ,# ট
'ঠেলাগাড়িতে ঠেলতে হয়' ,# ঠ
'ডালিম ফল ঝুলে গাছে' ,# ড
'ঢোলের শব্দ কানে বাজে' ,# ঢ
'হরিণটি দেখতে বেশ' ,# ণ
'তালা লাগায় সর্বশেষ' ,# ত
'থলের ভিতর বাজার ভরে' ,# থ
'দুধ পানে শক্তি বাড়ে' ,# দ
'ধান গাছের দেখ ছবি' ,# ধ
'নজরুল আমাদের জাতীয় কবি' ,# ন
'পুতুল নিয়ে খেলা করে' ,# প
'ফরিং বসেছে ঘাসের উপর' ,# ফ
'বাঘের গায়ে কালো ডোরা' ,# ব
'ভেড়ার শরীর লোমে ভরা' ,# ভ
'মা মেয়েকে যত্ন করে' ,# ম
'যাঁতা ঘোরে হাতের জোরে' ,# য
'রাজহাসটি ডাকছে দূরে' ,# র
'লাটিম নিয়ে খেলা করে' ,# ল
'শালিক পাখি ধান খায়' ,# শ
'ষাঁড় হয় শক্তিশালী' ,# ষ
'সাইকেল চলে পায়ের বলে' ,# স
'হাঁস ভাসে নদীর জলে', # হ
'ঘড়ি সঠিক সময় দেয়', # ড়
'আষাঢ় মাসে বৃষ্টি হয়', # ঢ়
'গয়না পড়ে বধু সাজে', # য়
'মৎস ধরে জাল ফেলে ', # ৎ
'মাংস মজা ভূনা খেলে', # ং
'দুঃখী মানুষ সেবা চায়', # ঃ
'চাঁদের আলোয় স্নিগ্ধা ছড়ায়' #  ঁ
]

print(len(alph), len(sentence))

model_path = 'd:/project/trained_model.h5'
class Frame:
    def __init__(self, root) -> None:
        self.root = root;
        self.model = load_model(model_path)
        self.Fmain = tk.Frame(self.root, bg='white');
        self.Fmain.place(x=0,y=0,relheight=1,relwidth=1)

        self.Fheader = tk.Frame(self.Fmain, bg=color['header-bg'])
        self.Fheader.place(x=0,y=0,relheight=0.1,relwidth=1)
        self.header = tk.Label(self.Fheader, bg=color['header-bg'], fg=color['header-fg'],
                text="Bangla Sentence Generation from Handwritten Alphabets",
                font=('Arial',18,'bold'))
        self.header.pack(pady=15)
        
        self.Ffooter = tk.Frame(self.Fmain, bg=color['footer-bg'])
        self.Ffooter.place(relx=0,rely=0.9,relwidth=1,relheight=0.1)
        
        self.Bsave = tk.Button(self.Ffooter, text='Generate', command=self.save_drawing)
        self.Bsave.place(relx=0.25,rely=0.3, relwidth=0.1)
        
        self.Bclear = tk.Button(self.Ffooter, text='Clear', command=self.clear_canvas)
        self.Bclear.place(relx=0.65,rely=0.3, relwidth=0.1)
        
        self.Fleft = tk.Frame(self.Fmain)
        self.Fleft.place(x=0,rely=0.1,relheight=0.8,relwidth=0.5)
        
        self.Bupload = tk.Button(self.Fleft, text='Upload', command=self.upload_image)
        self.Bupload.place(relx=0.4, rely=0.75, relwidth=0.2)
        
        self.Fcanvas = tk.Frame(self.Fleft,  bd=1, highlightthickness=1, highlightbackground='black')
        self.Fcanvas.place(relx=0.35,rely=0.3,relwidth=0.3,relheight=0.3)
        
        # Create an Image and ImageDraw object
        self.image = Image.new("RGB", (224,224), color="white")
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas = tk.Canvas(self.Fcanvas, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<B1-Motion>", self.paint)
        
        self.title = tk.Label(self.Fleft, bg='white',
                    fg='orange', text='Draw here', font=('arial',15,'bold'))
        self.title.place(relx=0.4, rely=0.65, relwidth=0.2)
    
        self.Fright = tk.Frame(self.Fmain, bg='white')
        self.Fright.place(relx=0.5,rely=0.1,relwidth=0.5,relheight=0.8)
        self.output = tk.Label(self.Fright, bg='white',
                    text='Draw & Save',font=('arial',50,'bold'))
        self.output.pack(fill=tk.BOTH,expand=True)
        
        # self.Ftop = tk.Frame(self.Fright)
        # self.Ftop.place(relx=0, rely=0, relwidth=1, relheight=0.4)
        
        # img= Image.open('ho.png')
        # image = tk.PhotoImage(img)
        # self.outputImage = tk.Label(self.Ftop, image=image)
        # self.outputImage.pack()
        
        self.outputSentence = tk.Label(self.Fright, bg='white', fg='green', font=('',20))
        self.outputSentence.place(x=0, rely=0.8, relheight=0.2, relwidth=1)
        
    def paint(self, event):
        # Get the current mouse position
        x, y = event.x, event.y

        # Draw a circle on the image
        radius = 5  # You can adjust the radius as needed
        x1, y1 = x - radius, y - radius
        x2, y2 = x + radius, y + radius
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

        # Update the canvas to display the drawing
        self.canvas.create_oval(x1, y1, x2, y2, fill="black")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
        if file_path:
            # Open the image file and resize it to fit the canvas
            uploaded_image = Image.open(file_path)
            resized_image = uploaded_image.resize((224, 224))

            # Display the uploaded image on the canvas
            self.image = resized_image
            self.draw = ImageDraw.Draw(self.image)

            # Create PhotoImage and keep a reference
            photo_image = ImageTk.PhotoImage(self.image)
            self.image_on_canvas = photo_image

            # Update the canvas to display the uploaded image
            self.canvas.create_image(0, 0, anchor="nw", image=self.image_on_canvas)

    def save_drawing(self):
        self.output.config(text='Waiting', fg='yellow')
        # Save the image directly to the current working directory
        self.image.save("drawing_snapshot.png", format="png")
        self.predict_drawing()
        
    def clear_canvas(self):
        # Clear the canvas by deleting all drawings
        self.canvas.delete("all")

        # Reset the internal Image and ImageDraw objects
        self.image = Image.new("RGB", (224,224), color="white")
        self.draw = ImageDraw.Draw(self.image)
        self.output.config(text='Draw & Save')
        self.outputSentence.config(text='')
        
    def predict_drawing(self):
        # Resize and preprocess the image for model input
        input_image = self.image.resize((224, 224))
        input_image.save("input_image.png",format='png')
        input_array = np.array(input_image) / 255.0
        input_array = input_array.reshape((1, 224, 224, 3))

        # Make a prediction
        prediction = self.model.predict(input_array)

        # Display the prediction (you can customize this part based on your model)
        predicted_class = np.argmax(prediction)
        print(f"Predicted Class: {predicted_class}")
        self.output.config(text=alph[predicted_class%len(alph)], fg='green')
        self.outputSentence.config(text=sentence[predicted_class%len(sentence)])
        
frame = Frame(root)

root.mainloop()