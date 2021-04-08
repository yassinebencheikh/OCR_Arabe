# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 21:06:59 2020

@author: Abdellah-Bencheikh
"""
import tkinter as tk

# class OCR_IHM
class OCR_IHM :
    """IHM de l'application 'répertoire téléphonique'."""
    def __init__(self):
        """Initialisateur/lanceur de la fenêtre de base"""
        self.root = tk.Tk()
        self.root.title("OCR Arabe")
        self.root.config(background="#d9d9d9")
        self.root.minsize(124, 1)
        self.root.maxsize(1362, 741)
        self.root.resizable(0, 0)
        #self.root.config(relief='groove')
        # self.root.config(relief=tk.RAISED, bd=3)
        self.makeWidgets()
        self.root.mainloop()
    def makeWidgets(self) :
        """Configure et positionne les widgets"""
        # créer une frame
        Frameprin = tk.Frame(self.root, height=1362, width=741)
        Frameprin.place(relx=0.0, rely=0.0)
        #Frameprin.configure(relief='groove')
        Frameprin.configure(borderwidth="2")
        Frameprin.configure(background="#c5dbfc")
        Frameprin.pack()
        font1 = "-family {Segoe UI} -size 9 -weight bold -slant "  \
            "italic"
        # LabelFrame Load image 
        Labframload = tk.LabelFrame(Frameprin, height=163, width=400)
        Labframload.place(in_= Frameprin, relx=0.068, rely=0.032)
        #Labframload.configure(relief='groove')
        Labframload.configure(font=font1)
        Labframload.configure(foreground="black")
        Labframload.configure(text='''Chargement d'une image''')
        Labframload.configure(background="#d9d9d9")
        #Labframload.grid(column=0,row=1,columnspan=2,sticky='EW')
        Labframload.pack()
        # label charger image
        Labload = tk.Label(Labframload, height=2, width=70)
        Labload.place(in_= Labframload, relx=0.173, rely=0.4)
        Labload.configure(background="#d9d9d9")
        Labload.configure(disabledforeground="#a3a3a3")
        Labload.configure(font="-family {Segoe UI} -size 9 -weight bold")
        Labload.configure(foreground="#000000")
        Labload.configure(text='''Charger l'image :''')
        Labload.pack()               
        # button rechercher
        Butrech = tk.Button(Labframload, command = self.actionRechercher)
        Butrech.place(in_= Labframload, relx=0.519, rely=0.4, height=24, width=167, bordermode='ignore')
        Butrech.configure(activebackground="#ececec")
        Butrech.configure(activeforeground="#000000")
        Butrech.configure(background="#64b1ff")
        Butrech.configure(disabledforeground="#a3a3a3")
        Butrech.configure(foreground="#000000")
        Butrech.configure(highlightbackground="#d9d9d9")
        Butrech.configure(highlightcolor="black")
        Butrech.configure(pady="0")
        Butrech.configure(text='''Rechercher image''')
        Butrech.pack()
        # LabelFrame Affichage
        LabFramAffich = tk.LabelFrame(Frameprin, height=431, width=657)
        LabFramAffich.place(relx=0.068, rely=0.208)
        LabFramAffich.configure(relief='groove')
        LabFramAffich.configure(font=font1)
        LabFramAffich.configure(foreground="black")
        LabFramAffich.configure(text='''Affichage''')
        LabFramAffich.configure(background="#d9d9d9")
        LabFramAffich.pack()
        # label charger image
        self.Cadre = tk.Label(LabFramAffich)
        self.Cadre.place(in_= LabFramAffich, relx=0.038, rely=0.111, height=431, width=650)
        self.Cadre.configure(background="#d9d9d9")
        self.Cadre.configure(disabledforeground="#a3a3a3")
        self.Cadre.configure(foreground="#000000")
        self.Cadre.configure(text='''''')
        self.Cadre.pack()
        # Button Convertir
        btnConverti = tk.Button(Frameprin, command = self.actionConverti)
        btnConverti.place(relx=0.597, rely=0.16, height=24, width=140)
        btnConverti.configure(activebackground="#ececec")
        btnConverti.configure(activeforeground="#000000")
        btnConverti.configure(background="#64b1ff")
        btnConverti.configure(disabledforeground="#a3a3a3")
        btnConverti.configure(foreground="#000000")
        btnConverti.configure(highlightbackground="#d9d9d9")
        btnConverti.configure(highlightcolor="black")
        btnConverti.configure(pady="0")
        btnConverti.configure(text='''Convertir''')
        btnConverti.pack()
        # LabelFrame Texte
        self.Labelframe1 = tk.LabelFrame(Frameprin, height=100, width=100)
        self.Labelframe1.place(relx=0.068, rely=0.545)
        self.Labelframe1.configure(relief='groove')
        self.Labelframe1.configure(font=font1)
        self.Labelframe1.configure(foreground="black")
        self.Labelframe1.configure(text='''Texte''')
        self.Labelframe1.configure(background="#d9d9d9")
        self.Labelframe1.pack()
        # Texte 
        # self.texte = tk.Text(self.Labelframe1, height=15, width=70)  
        # self.scroll = tk.Scrollbar(self.Labelframe1, command =self.texte.yview)  
        # self.texte.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        # self.scroll.grid(row=0, column=1, sticky='nsew')
        # self.texte.configure(yscrollcommand = self.scroll.set)
        # self.texte.place(in_= self.Labelframe1, relx=0.038, rely=0.089)
        # self.texte.configure(background="white")
        # self.texte.configure(font="-family {Segoe UI} -size 9 -weight bold")
        # self.texte.configure(foreground="black")
        # self.texte.configure(highlightbackground="#d9d9d9")
        # self.texte.configure(highlightcolor="black")
        # self.texte.configure(insertbackground="black")
        # self.texte.configure(selectbackground="#c4c4c4")
        # self.texte.configure(selectforeground="black")
        # self.texte.configure(wrap="word")
        # self.scroll.pack()
        # self.texte.pack()
        
        
        self.texte = tk.Text(self.Labelframe1, height=15, width=70)  
        self.scroll = tk.Scrollbar(self.Labelframe1, command =self.texte.yview)  
        self.texte.configure(yscrollcommand = self.scroll.set)
        self.scroll.pack(side='right', fill='y')
        
        # self.texte.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        # self.scroll.grid(row=0, column=1, sticky='nsew')
        self.texte.place(in_= self.Labelframe1, relx=0.038, rely=0.089)
        self.texte.configure(background="white")
        self.texte.configure(font="-family {Segoe UI} -size 9 -weight bold")
        self.texte.configure(foreground="black")
        self.texte.configure(highlightbackground="#d9d9d9")
        self.texte.configure(highlightcolor="black")
        self.texte.configure(insertbackground="black")
        self.texte.configure(selectbackground="#c4c4c4")
        self.texte.configure(selectforeground="black")
        self.texte.configure(wrap="word")
        # self.scroll.pack()
        # self.texte.pack()
        self.texte.pack(fill='both', expand=1)
        

        # button enregistrer
        ButEnregistrer = tk.Button(Frameprin, command = self.actionSave)
        ButEnregistrer.place(relx=0.392, rely=0.929, height=24, width=137)
        ButEnregistrer.configure(activebackground="#ececec")
        ButEnregistrer.configure(activeforeground="#000000")
        ButEnregistrer.configure(background="#64b1ff")
        ButEnregistrer.configure(disabledforeground="#a3a3a3")
        ButEnregistrer.configure(foreground="#000000")
        ButEnregistrer.configure(highlightbackground="#d9d9d9")
        ButEnregistrer.configure(highlightcolor="black")
        ButEnregistrer.configure(pady="0")
        ButEnregistrer.configure(text='''Enregistrer texte''')
        ButEnregistrer.pack()

    def actionConverti(self) :
        pass

    def actionRechercher(self) :
        pass

    def actionSave(self) :
        pass


if __name__ == '__main__' :
    # instancie l'IHM
    app = OCR_IHM()