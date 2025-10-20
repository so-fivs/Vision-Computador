#include <stdint.h>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

String face_cascade_name = "../data/haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

int main()
{
    //-- Load the cascades
    if(!face_cascade.load(face_cascade_name))
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };
    if(!eyes_cascade.load(eyes_cascade_name))
    {
        cout << "--(!)Error loading eyes cascade\n";
        return -1;
    };

    //::========================================================================================================
    //::======================================== PUNTO 1 =======================================================
    //::========================================================================================================
    // Realizar la detección del objeto de interés (rostro) por el método Haar en book.png
    // Mostrar solo el ROI del rostro detectado
    //::========================================================================================================
    
    cout << "\n========== PUNTO 1: Deteccion de Rostro (Haar Cascade) ==========" << endl;
    cout << "Cargando imagen book.png..." << endl;
    
    Mat img_book = imread("../data/book.png");
    if(img_book.empty())
    {
        cout << "Error: No se pudo cargar book.png" << endl;
        return -1;
    }
    
    Mat img_book_gray;
    cvtColor(img_book, img_book_gray, COLOR_BGR2GRAY);
    equalizeHist(img_book_gray, img_book_gray);
    
    // Detectar rostros con Haar Cascade
    std::vector<Rect> faces;
    face_cascade.detectMultiScale(img_book_gray, faces);
    
    if(faces.empty())
    {
        cout << "No se detectó ningún rostro en book.png" << endl;
        return -1;
    }
    
    // Extraer ROI del primer rostro detectado
    Rect face_rect = faces[0];
    Mat roi_face_global = img_book_gray(face_rect).clone();
    
    cout << "Rostro detectado - ROI extraído: [" << face_rect.width << "x" << face_rect.height << "]" << endl;
    
    // Mostrar SOLO el ROI del rostro
    Mat roi_face_color;
    cvtColor(roi_face_global, roi_face_color, COLOR_GRAY2BGR);
    imshow("PUNTO 1: Objeto Detallado (ROI)", roi_face_color);
    
    //::======================================== FIN PUNTO 1 ===================================================
    
    
    
    //::========================================================================================================
    //::======================================== PUNTO 2 =======================================================
    //::========================================================================================================
    // Extraer las características visuales sobre la sub-imagen o región de interés ROI 
    // del rostro detectado usando los métodos BRISK y FREAK
    //::========================================================================================================
    
    cout << "\n========== PUNTO 2: Extraccion Caracteristicas Visuales ROI ==========" << endl;
    
    // Crear detectores y descriptores
    Ptr<BRISK> brisk_detector = BRISK::create(30, 3, 1.0f);
    Ptr<FREAK> freak_descriptor = FREAK::create();
    
    // --- BRISK: Detector y Descriptor sobre ROI ---
    std::vector<KeyPoint> keypoints_roi_brisk;
    Mat descriptors_roi_brisk;
    brisk_detector->detectAndCompute(roi_face_global, noArray(), 
                                     keypoints_roi_brisk, descriptors_roi_brisk);
    
    cout << "BRISK - Keypoints detectados en ROI: " << keypoints_roi_brisk.size() << endl;
    
    // --- FREAK: Usar BRISK como detector, FREAK como descriptor ---
    std::vector<KeyPoint> keypoints_roi_freak;
    Mat descriptors_roi_freak;
    brisk_detector->detect(roi_face_global, keypoints_roi_freak);
    freak_descriptor->compute(roi_face_global, keypoints_roi_freak, descriptors_roi_freak);
    
    cout << "FREAK - Keypoints detectados en ROI: " << keypoints_roi_freak.size() << endl;
    
    //::======================================== FIN PUNTO 2 ===================================================
    
    
    
    //::========================================================================================================
    //::======================================== PUNTO 3 =======================================================
    //::========================================================================================================
    // Extraer las características visuales de toda la imagen, frame a frame en el video 
    // "blais.mp4" usando los métodos BRISK y FREAK
    // Además, detectar rostro y ojos en cada frame del video
    //::========================================================================================================
    
    cout << "\n========== PUNTO 3: Extraccion Caracteristicas en Video ==========" << endl;
    
    VideoCapture capture("../data/blais.mp4");
    if(!capture.isOpened())
    {
        cout << "Error: No se pudo abrir el video blais.mp4" << endl;
        return 0;
    }
    
    // Crear matchers para PUNTO 4
    BFMatcher matcher_brisk(NORM_HAMMING);
    BFMatcher matcher_freak(NORM_HAMMING);
    
    Mat img_scene, img_scene_gray;
    int frame_count = 0;
    
    cout << "Procesando video blais.mp4... (Presiona ESC para salir)" << endl;
    
    while(true)
    {
        capture >> img_scene;
        
        if(img_scene.empty())
            break;
        
        cvtColor(img_scene, img_scene_gray, COLOR_BGR2GRAY);
        equalizeHist(img_scene_gray, img_scene_gray);
        
        //--- Detección de rostro y ojos en el video ---
        Mat img_detection = img_scene.clone();
        
        std::vector<Rect> faces_video;
        face_cascade.detectMultiScale(img_scene_gray, faces_video);
        
        for(size_t i = 0; i < faces_video.size(); i++)
        {
            // Dibujar rectángulo del rostro
            rectangle(img_detection, faces_video[i], Scalar(255, 0, 255), 3);
            
            // Detectar y dibujar ojos
            Mat faceROI_video = img_scene_gray(faces_video[i]);
            std::vector<Rect> eyes_video;
            eyes_cascade.detectMultiScale(faceROI_video, eyes_video);
            
            for(size_t j = 0; j < eyes_video.size(); j++)
            {
                Point eye_center(faces_video[i].x + eyes_video[j].x + eyes_video[j].width/2, 
                                faces_video[i].y + eyes_video[j].y + eyes_video[j].height/2);
                int radius = cvRound((eyes_video[j].width + eyes_video[j].height)*0.25);
                circle(img_detection, eye_center, radius, Scalar(255, 0, 0), 4);
            }
        }
        
        imshow("PUNTO 3: Deteccion Rostro y Ojos en Video", img_detection);
        
        
        //::====================================================================================================
        //::======================================== PUNTO 4 ===================================================
        //::====================================================================================================
        // Buscar las características visuales BRISK y FREAK del rostro detectado frame a frame
        // en el video "blais.mp4". Realizar matching con el método de fuerza bruta y graficar
        // con el método de homografía
        //::====================================================================================================
        
        //--- BRISK: Extracción de características del frame completo ---
        std::vector<KeyPoint> keypoints_scene_brisk;
        Mat descriptors_scene_brisk;
        
        brisk_detector->detectAndCompute(img_scene_gray, noArray(), 
                                        keypoints_scene_brisk, descriptors_scene_brisk);
        
        // Matching BRISK con k=2 para ratio test de Lowe
        std::vector<std::vector<DMatch>> knn_matches_brisk;
        if(!descriptors_scene_brisk.empty() && !descriptors_roi_brisk.empty())
        {
            matcher_brisk.knnMatch(descriptors_roi_brisk, descriptors_scene_brisk, 
                                   knn_matches_brisk, 2);
        }
        
        // Filtrar buenos matches BRISK (ratio test)
        std::vector<DMatch> good_matches_brisk;
        const float ratio_thresh = 0.85f;
        for(size_t i = 0; i < knn_matches_brisk.size(); i++)
        {
            if(knn_matches_brisk[i].size() >= 2 && 
               knn_matches_brisk[i][0].distance < ratio_thresh * knn_matches_brisk[i][1].distance)
            {
                good_matches_brisk.push_back(knn_matches_brisk[i][0]);
            }
        }
        
        // Visualizar matches BRISK
        Mat img_matches_brisk;
        drawMatches(roi_face_global, keypoints_roi_brisk, 
                   img_scene, keypoints_scene_brisk,
                   good_matches_brisk, img_matches_brisk, 
                   Scalar::all(-1), Scalar::all(-1),
                   vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        
        imshow("PUNTO 4: Matches BRISK", img_matches_brisk);
        
        
        //--- FREAK: Extracción de características del frame completo ---
        std::vector<KeyPoint> keypoints_scene_freak;
        Mat descriptors_scene_freak;
        
        brisk_detector->detect(img_scene_gray, keypoints_scene_freak);
        freak_descriptor->compute(img_scene_gray, keypoints_scene_freak, 
                                 descriptors_scene_freak);
        
        // Matching FREAK con k=2 para ratio test de Lowe
        std::vector<std::vector<DMatch>> knn_matches_freak;
        if(!descriptors_scene_freak.empty() && !descriptors_roi_freak.empty())
        {
            matcher_freak.knnMatch(descriptors_roi_freak, descriptors_scene_freak, 
                                   knn_matches_freak, 2);
        }
        
        // Filtrar buenos matches FREAK (ratio test)
        std::vector<DMatch> good_matches_freak;
        for(size_t i = 0; i < knn_matches_freak.size(); i++)
        {
            if(knn_matches_freak[i].size() >= 2 && 
               knn_matches_freak[i][0].distance < ratio_thresh * knn_matches_freak[i][1].distance)
            {
                good_matches_freak.push_back(knn_matches_freak[i][0]);
            }
        }
        
        // Visualizar matches FREAK
        Mat img_matches_freak;
        drawMatches(roi_face_global, keypoints_roi_freak, 
                   img_scene, keypoints_scene_freak,
                   good_matches_freak, img_matches_freak, 
                   Scalar::all(-1), Scalar::all(-1),
                   vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        
        imshow("PUNTO 4: Matches FREAK", img_matches_freak);
        
        //::======================================== FIN PUNTO 4 ===============================================
        
        frame_count++;
        
        // Control: ESC para salir
        int key = waitKey(30);
        if(key == 27)
            break;
    }
    
    //::======================================== FIN PUNTO 3 ===================================================
    

    //::========================================================================================================
    //::======================================== RESUMEN FINAL =================================================
    //::========================================================================================================
    
    cout << "\n========== RESUMEN DE EJECUCION ==========" << endl;
    cout << "Total de frames procesados: " << frame_count << endl;
    cout << "BRISK - Keypoints en ROI: " << keypoints_roi_brisk.size() << endl;
    cout << "FREAK - Keypoints en ROI: " << keypoints_roi_freak.size() << endl;
    cout << "\n=== PROGRAMA FINALIZADO EXITOSAMENTE ===" << endl;
    
    capture.release();
    destroyAllWindows();
    
    return 0;
}