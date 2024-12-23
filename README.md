# **Car Visualizer with GAN and Object Detection**

This project leverages advanced AI techniques, including Generative Adversarial Networks (GANs) and object detection, to create a robust system for generating and visualizing cars. It combines cutting-edge technology to provide a powerful tool for automotive visualization.

---

## **Features**
- **Car Image Generation**: Generate images of cars using GANs.
- **Object Detection**: Utilize YOLO for precise car detection and cropping.
- **AI Integration**: Combines GANs and YOLO for enhanced image processing capabilities.

---

## **Setup and Installation**
1. **Clone the Repository**:  
   ```bash
   git clone <repository-url>
   cd car-visualizer
   ```

2. **Create a Virtual Environment**:  
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Dataset**:  
   - Ensure cropped car images are available in the `cropped_cars/` directory.

---

## **Usage**
### Train the GAN:  
To train the GAN and generate car images:  
```bash
python train_gan.py
```  

### Visualize Generated Cars:  
Generated images are displayed every 10 epochs. You can modify the interval in the script.

---

## **Dataset**
The dataset for this project consists of cropped images of cars, preprocessed for GAN training. Ensure the dataset is placed in the `cropped_cars/` directory.

---

## **Technical Notes**
- **GAN Implementation**: Includes custom Generator and Discriminator architectures for generating realistic car images.
- **Object Detection**: YOLOv8 is used for detecting and cropping cars from raw images.
- **Dependencies**: Refer to `requirements.txt` for a complete list of required Python libraries.

---


## **Contributing**
Contributions are welcome! To contribute:  
1. Fork the repository.  
2. Create a feature branch (`git checkout -b feature-name`).  
3. Commit your changes (`git commit -m 'Add feature'`).  
4. Push to the branch (`git push origin feature-name`).  
5. Open a pull request.  

---


## **Contact**
For questions or collaboration:  
**Etenesh Gishamo**  
- Email: etenesh4good@gmail.com
- LinkedIn: https://www.linkedin.com/in/etenesh-gishamo-1b13a2272/

---

