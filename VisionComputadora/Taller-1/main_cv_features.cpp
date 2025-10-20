#include <stdint.h>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

// Includes de OpenCV
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

using std::cout;
using std::endl;

//----------------MEJORES MATCHES---------------------------------

void FindGoodMatches(
    const Mat& descriptors_object,
    const Mat& descriptors_scene,
    std::vector<DMatch>& good_matches,
    int matcher_type 
)
{
    if (descriptors_object.empty() || descriptors_scene.empty()) {
        return; 
    }

    Ptr<DescriptorMatcher> matcher;
    if (matcher_type == 1) 
    {
        // Hamming para descriptores binarios (BRISK, FREAK, BRIEF, ORB)
        matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
    }
    else 
    {
        // L2 para descriptores flotantes (SIFT, SURF)
        matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    }
    
    std::vector< std::vector<DMatch> > matches_knn;
    const float ratio_thresh = 0.85f; 

    // Try-catch dentro de FindGoodMatches
    try {
        matcher->knnMatch(descriptors_object, descriptors_scene, matches_knn, 2);
    } catch (const cv::Exception& e) {
        return;
    }


    for (size_t i = 0; i < matches_knn.size(); i++)
    {
        if (matches_knn[i].size() >= 2)
        {
            if (matches_knn[i][0].distance < ratio_thresh * matches_knn[i][1].distance)
            {
                  good_matches.push_back(matches_knn[i][0]);
            }
        }
    }
}

//----------------VISUALIZACION Y HOMOGRAFIA---------------------------------
Mat DrawHomography(
    const Mat& fto_objeto, 
    const Mat& fto_scene, 
    const std::vector<KeyPoint>& keypoints_obj, 
    const std::vector<KeyPoint>& keypoints_scene, 
    const std::vector<DMatch>& good_matches,
    const std::string& title)
{
    std::vector<Point2f> obj_points;
    std::vector<Point2f> scene_points;
    Mat img_matches;
    size_t num_matches = good_matches.size();

    if (num_matches < 4) // Mínimo 4 puntos para Homografía
    {
        cout << title << " -> Combinación no compatible / Insuficientes matches (" << num_matches << ")" << endl;
        return Mat();
    }
    
    for( size_t i = 0; i < num_matches; i++ )
    {
        int queryIdx = good_matches[i].queryIdx;
        int trainIdx = good_matches[i].trainIdx;
        
        if (queryIdx >= 0 && queryIdx < (int)keypoints_obj.size() &&
            trainIdx >= 0 && trainIdx < (int)keypoints_scene.size())
        {
            obj_points.push_back( keypoints_obj[queryIdx].pt );
            scene_points.push_back( keypoints_scene[trainIdx].pt );
        }
    }

    if (obj_points.size() < 4) // Mínimo 4 puntos para Homografía
    {
        cout << title << " -> Combinación no compatible / Puntos válidos insuficientes (" << obj_points.size() << ")" << endl;
        return Mat();
    }

    // Homografía si hay 10 o más matches
    if (obj_points.size() >= 10) 
    {
        // TRY-CATCH para proteger la llamada inestable de drawMatches/findHomography
        try {
            Mat H = findHomography( obj_points, scene_points, RANSAC );

            // Si H es nulo o inválido, findHomography falló.
            if (H.empty()) {
                cout << title << " -> ERROR: Homografía fallida. Puntos RANSAC insuficientes." << endl;
                return Mat();
            }

            std::vector<Point2f> obj_corners(4);
            obj_corners[0] = Point2f(0, 0); 
            obj_corners[1] = Point2f((float)fto_objeto.cols, 0);
            obj_corners[2] = Point2f((float)fto_objeto.cols, (float)fto_objeto.rows);
            obj_corners[3] = Point2f(0, (float)fto_objeto.rows);

            std::vector<Point2f> scene_corners(4);
            perspectiveTransform(obj_corners, scene_corners, H);

            drawMatches(fto_objeto, keypoints_obj, fto_scene, keypoints_scene, 
                        good_matches, img_matches, 
                        Scalar::all(-1), Scalar::all(-1), 
                        std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            if (img_matches.channels() < 3) 
                cvtColor(img_matches, img_matches, COLOR_GRAY2BGR);
            
            // Dibujar el contorno en la escena
            Point2f offset((float)fto_objeto.cols, 0.0f);
            line(img_matches, scene_corners[0] + offset, scene_corners[1] + offset, Scalar(0, 255, 0), 4);
            line(img_matches, scene_corners[1] + offset, scene_corners[2] + offset, Scalar(0, 255, 0), 4);
            line(img_matches, scene_corners[2] + offset, scene_corners[3] + offset, Scalar(0, 255, 0), 4);
            line(img_matches, scene_corners[3] + offset, scene_corners[0] + offset, Scalar(0, 255, 0), 4);
            
        } catch (const cv::Exception& e) {
            // Este catch maneja el error de resize/drawMatches
            cout << title << " -> ERROR interno de OpenCV al dibujar/Homografía. Ignorado: " << e.what() << endl;
            return Mat();
        }
        
    } else {
        cout << title << " -> Solo visualización de matches (Cantidad: " << obj_points.size() << "). Min 10 requeridos para Homografía." << endl;
        
        // TRY-CATCH para proteger la llamada inestable de drawMatches (solo visualización)
        try {
            drawMatches(fto_objeto, keypoints_obj, fto_scene, keypoints_scene, 
                        good_matches, img_matches, 
                        Scalar::all(-1), Scalar::all(-1), 
                        std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        } catch (const cv::Exception& e) {
            cout << title << " -> ERROR interno de OpenCV al dibujar matches. Ignorado: " << e.what() << endl;
            return Mat();
        }
    }

    return img_matches;
}

int main()
{
    string nom_objeto="/Users/sofiavelandiasierra/Documents/Vision-projects/VisionComputadora/Taller-1/box.png";
    string nom_secene="/Users/sofiavelandiasierra/Documents/Vision-projects/VisionComputadora/Taller-1/box_in_scene.png";

    Mat fto_objeto = imread(nom_objeto , IMREAD_GRAYSCALE );
    Mat fto_scene = imread(nom_secene , IMREAD_GRAYSCALE );

    if ( (fto_objeto.empty()) || (fto_scene.empty()) ) {
        cout << "Could not open or find the image!\n" << endl;
         return -1;
    }

//---------------DECLARACION DE ALGORITMOS ----------------------------------
   
   // Parámetros y creación de descriptores y detector
   Ptr<SIFT> feat_sift = SIFT::create(0, 10, 0.004, 10, 1.6); 
   Ptr<SURF> feat_surf = SURF::create(300, 4, 4, false, false); 
   Ptr<BRISK> feat_brisk = BRISK::create(300, 20, 1.0f);
   Ptr<ORB> feat_orb = ORB::create(500, 1.2f, 12, 31, 0, 2, ORB::HARRIS_SCORE, 20, 20); 
   Ptr<FREAK> feat_freak = FREAK::create(true, true, 16.0f, 10);
   Ptr<xfeatures2d::BriefDescriptorExtractor> feat_brief = xfeatures2d::BriefDescriptorExtractor::create(32, true);
   Ptr<FastFeatureDetector> detector_fast = FastFeatureDetector::create(10, false, FastFeatureDetector::TYPE_9_16);

// -------------------------------------------------------------------------------------
// --------------- DETECCIÓN DE KEYPOINTS SIFT (Base para pruebas 1-6) -----------------
// -------------------------------------------------------------------------------------

std::vector<KeyPoint> keypoints_sift_objeto, keypoints_sift_scene;
feat_sift->detect(fto_objeto, keypoints_sift_objeto);
feat_sift->detect(fto_scene, keypoints_sift_scene);

cout << "\nSIFT KeyPoints Objeto: " << keypoints_sift_objeto.size() << endl;
cout << "SIFT KeyPoints Escena: " << keypoints_sift_scene.size() << endl;

// -------------------------------------------------------------------------------------
// --------------- CÁLCULO DE DESCRIPTORES CON BASE SIFT KEYPOINTS ---------------------
// -------------------------------------------------------------------------------------

// Declaración de Mats para descriptores basados en SIFT KP
Mat D_sift_sift_obj, D_sift_surf_obj, D_sift_brisk_obj, D_sift_freak_obj, D_sift_brief_obj, D_sift_orb_obj;
Mat D_sift_sift_scene, D_sift_surf_scene, D_sift_brisk_scene, D_sift_freak_scene, D_sift_brief_scene, D_sift_orb_scene;

// Cálculo de descriptores para el OBJETO
feat_sift->compute(fto_objeto, keypoints_sift_objeto, D_sift_sift_obj);
feat_surf->compute(fto_objeto, keypoints_sift_objeto, D_sift_surf_obj);
feat_brisk->compute(fto_objeto, keypoints_sift_objeto, D_sift_brisk_obj);
feat_freak->compute(fto_objeto, keypoints_sift_objeto, D_sift_freak_obj);
feat_brief->compute(fto_objeto, keypoints_sift_objeto, D_sift_brief_obj);
feat_orb->compute(fto_objeto, keypoints_sift_objeto, D_sift_orb_obj);

// Cálculo de descriptores para la ESCENA
feat_sift->compute(fto_scene, keypoints_sift_scene, D_sift_sift_scene);
feat_surf->compute(fto_scene, keypoints_sift_scene, D_sift_surf_scene);
feat_brisk->compute(fto_scene, keypoints_sift_scene, D_sift_brisk_scene);
feat_freak->compute(fto_scene, keypoints_sift_scene, D_sift_freak_scene);
feat_brief->compute(fto_scene, keypoints_sift_scene, D_sift_brief_scene);
feat_orb->compute(fto_scene, keypoints_sift_scene, D_sift_orb_scene);


//--- 1. SIFT-SIFT (L2) ----------------------------------------------------
std::vector<DMatch> M_sift_sift;
Mat img_sift_sift;
FindGoodMatches(D_sift_sift_obj, D_sift_sift_scene, M_sift_sift, 0); // 0 = L2
cout << "1. SIFT-SIFT Good Matches: " << M_sift_sift.size() << endl;
img_sift_sift = DrawHomography(fto_objeto, fto_scene, keypoints_sift_objeto, keypoints_sift_scene, M_sift_sift, "1. SIFT-SIFT");

//--- 2. SURF-SURF (L2) - SIFT KP Base ---------------------------------------
std::vector<DMatch> M_sift_surf;
Mat img_sift_surf;
FindGoodMatches(D_sift_surf_obj, D_sift_surf_scene, M_sift_surf, 0); // 0 = L2
cout << "2. SIFT-SURF Good Matches: " << M_sift_surf.size() << endl;
img_sift_surf = DrawHomography(fto_objeto, fto_scene, keypoints_sift_objeto, keypoints_sift_scene, M_sift_surf, "2. SIFT-SURF");

//--- 3. BRISK-BRISK (Hamming) - SIFT KP Base --------------------------------
std::vector<DMatch> M_sift_brisk;
Mat img_sift_brisk;
FindGoodMatches(D_sift_brisk_obj, D_sift_brisk_scene, M_sift_brisk, 1); // 1 = Hamming
cout << "3. SIFT-BRISK Good Matches: " << M_sift_brisk.size() << endl;
img_sift_brisk = DrawHomography(fto_objeto, fto_scene, keypoints_sift_objeto, keypoints_sift_scene, M_sift_brisk, "3. SIFT-BRISK");

//--- 4. FREAK-FREAK (Hamming) - SIFT KP Base --------------------------------
std::vector<DMatch> M_sift_freak;
Mat img_sift_freak;
FindGoodMatches(D_sift_freak_obj, D_sift_freak_scene, M_sift_freak, 1); // 1 = Hamming
cout << "4. SIFT-FREAK Good Matches: " << M_sift_freak.size() << endl;
img_sift_freak = DrawHomography(fto_objeto, fto_scene, keypoints_sift_objeto, keypoints_sift_scene, M_sift_freak, "4. SIFT-FREAK");

//--- 5. BRIEF-BRIEF (Hamming) - SIFT KP Base --------------------------------
std::vector<DMatch> M_sift_brief;
Mat img_sift_brief;
FindGoodMatches(D_sift_brief_obj, D_sift_brief_scene, M_sift_brief, 1); // 1 = Hamming
cout << "5. SIFT-BRIEF Good Matches: " << M_sift_brief.size() << endl;
img_sift_brief = DrawHomography(fto_objeto, fto_scene, keypoints_sift_objeto, keypoints_sift_scene, M_sift_brief, "5. SIFT-BRIEF");

//--- 6. ORB-ORB (Hamming) - SIFT KP Base (CORREGIDO: Usa Norma 1 para ORB) ---
std::vector<DMatch> M_sift_orb;
Mat img_sift_orb;
FindGoodMatches(D_sift_orb_obj, D_sift_orb_scene, M_sift_orb, 1); // 1 = Hamming (CORREGIDO)
cout << "6. SIFT-ORB Good Matches: " << M_sift_orb.size() << endl;
img_sift_orb = DrawHomography(fto_objeto, fto_scene, keypoints_sift_objeto, keypoints_sift_scene, M_sift_orb, "6. SIFT-ORB");
 
//--------------------------------------------------------------------------------------------------
//--------------- DETECCIÓN DE KEYPOINTS FAST (Base para pruebas 7-12) -----------------------------

std::vector<KeyPoint> keypoints_fast_objeto, keypoints_fast_scene;
detector_fast->detect(fto_objeto, keypoints_fast_objeto);
detector_fast->detect(fto_scene, keypoints_fast_scene);

cout << "\nFAST KeyPoints Objeto: " << keypoints_fast_objeto.size() << endl;
cout << "FAST KeyPoints Escena: " << keypoints_fast_scene.size() << endl;

// -------------------------------------------------------------------------------------
// --------------- CÁLCULO DE DESCRIPTORES CON BASE FAST KEYPOINTS ---------------------
// -------------------------------------------------------------------------------------

// Declaración de Mats para descriptores basados en FAST KP
Mat D_fast_sift_obj, D_fast_brisk_obj, D_fast_surf_obj, D_fast_orb_obj, D_fast_brief_obj, D_fast_freak_obj;
Mat D_fast_sift_scene, D_fast_brisk_scene, D_fast_surf_scene, D_fast_orb_scene, D_fast_brief_scene, D_fast_freak_scene;

// Cálculo de descriptores para el OBJETO
feat_sift->compute(fto_objeto, keypoints_fast_objeto, D_fast_sift_obj);
feat_brisk->compute(fto_objeto, keypoints_fast_objeto, D_fast_brisk_obj);
feat_surf->compute(fto_objeto, keypoints_fast_objeto, D_fast_surf_obj);
feat_orb->compute(fto_objeto, keypoints_fast_objeto, D_fast_orb_obj);
feat_brief->compute(fto_objeto, keypoints_fast_objeto, D_fast_brief_obj);
feat_freak->compute(fto_objeto, keypoints_fast_objeto, D_fast_freak_obj);

// Cálculo de descriptores para la ESCENA
feat_sift->compute(fto_scene, keypoints_fast_scene, D_fast_sift_scene);
feat_brisk->compute(fto_scene, keypoints_fast_scene, D_fast_brisk_scene);
feat_surf->compute(fto_scene, keypoints_fast_scene, D_fast_surf_scene);
feat_orb->compute(fto_scene, keypoints_fast_scene, D_fast_orb_scene);
feat_brief->compute(fto_scene, keypoints_fast_scene, D_fast_brief_scene);
feat_freak->compute(fto_scene, keypoints_fast_scene, D_fast_freak_scene);


//--- 7. SIFT-SIFT (L2) - FAST KP Base ---------------------------------------
std::vector<DMatch> M_fast_sift;
Mat img_fast_sift;
FindGoodMatches(D_fast_sift_obj, D_fast_sift_scene, M_fast_sift, 0); // 0 = L2
cout << "7. FAST-SIFT Good Matches: " << M_fast_sift.size() << endl;
img_fast_sift = DrawHomography(fto_objeto, fto_scene, keypoints_fast_objeto, keypoints_fast_scene, M_fast_sift, "7. FAST-SIFT");

//--- 8. BRISK-BRISK (Hamming) - FAST KP Base --------------------------------
std::vector<DMatch> M_fast_brisk;
Mat img_fast_brisk;
FindGoodMatches(D_fast_brisk_obj, D_fast_brisk_scene, M_fast_brisk, 1); // 1 = Hamming
cout << "8. FAST-BRISK Good Matches: " << M_fast_brisk.size() << endl;
img_fast_brisk = DrawHomography(fto_objeto, fto_scene, keypoints_fast_objeto, keypoints_fast_scene, M_fast_brisk, "8. FAST-BRISK");

//--- 9. SURF-SURF (L2) - FAST KP Base ---------------------------------------
std::vector<DMatch> M_fast_surf;
Mat img_fast_surf;
FindGoodMatches(D_fast_surf_obj, D_fast_surf_scene, M_fast_surf, 0); // 0 = L2
cout << "9. FAST-SURF Good Matches: " << M_fast_surf.size() << endl;
img_fast_surf = DrawHomography(fto_objeto, fto_scene, keypoints_fast_objeto, keypoints_fast_scene, M_fast_surf, "9. FAST-SURF");

//--- 10. ORB-ORB (Hamming) - FAST KP Base -----------------------------------
std::vector<DMatch> M_fast_orb;
Mat img_fast_orb;
FindGoodMatches(D_fast_orb_obj, D_fast_orb_scene, M_fast_orb, 1); // 1 = Hamming
cout << "10. FAST-ORB Good Matches: " << M_fast_orb.size() << endl;
img_fast_orb = DrawHomography(fto_objeto, fto_scene, keypoints_fast_objeto, keypoints_fast_scene, M_fast_orb, "10. FAST-ORB");

//--- 11. BRIEF-BRIEF (Hamming) - FAST KP Base -------------------------------
std::vector<DMatch> M_fast_brief;
Mat img_fast_brief;
FindGoodMatches(D_fast_brief_obj, D_fast_brief_scene, M_fast_brief, 1); // 1 = Hamming
cout << "11. FAST-BRIEF Good Matches: " << M_fast_brief.size() << endl;
img_fast_brief = DrawHomography(fto_objeto, fto_scene, keypoints_fast_objeto, keypoints_fast_scene, M_fast_brief, "11. FAST-BRIEF");

//--- 12. FREAK-FREAK (Hamming) - FAST KP Base -------------------------------
std::vector<DMatch> M_fast_freak;
Mat img_fast_freak;
FindGoodMatches(D_fast_freak_obj, D_fast_freak_scene, M_fast_freak, 1); // 1 = Hamming
cout << "12. FAST-FREAK Good Matches: " << M_fast_freak.size() << endl;
img_fast_freak = DrawHomography(fto_objeto, fto_scene, keypoints_fast_objeto, keypoints_fast_scene, M_fast_freak, "12. FAST-FREAK");


//--------------------------------------------------------------------------------------------------
//--------------- DETECCIÓN DE KEYPOINTS SURF (Base para pruebas 20-25) ----------------------------

std::vector<KeyPoint> keypoints_surf_objeto, keypoints_surf_scene;
feat_surf->detect(fto_objeto, keypoints_surf_objeto);
feat_surf->detect(fto_scene, keypoints_surf_scene);

cout << "\nSURF KeyPoints Objeto: " << keypoints_surf_objeto.size() << endl;
cout << "SURF KeyPoints Escena: " << keypoints_surf_scene.size() << endl;

// -------------------------------------------------------------------------------------
// --------------- CÁLCULO DE DESCRIPTORES CON BASE SURF KEYPOINTS ---------------------
// -------------------------------------------------------------------------------------

// Declaración de Mats para descriptores basados en SURF KP
Mat D_surf_sift_obj, D_surf_surf_obj, D_surf_brisk_obj, D_surf_freak_obj, D_surf_brief_obj, D_surf_orb_obj;
Mat D_surf_sift_scene, D_surf_surf_scene, D_surf_brisk_scene, D_surf_freak_scene, D_surf_brief_scene, D_surf_orb_scene;

// Cálculo de descriptores para el OBJETO
feat_sift->compute(fto_objeto, keypoints_surf_objeto, D_surf_sift_obj);
feat_surf->compute(fto_objeto, keypoints_surf_objeto, D_surf_surf_obj);
feat_brisk->compute(fto_objeto, keypoints_surf_objeto, D_surf_brisk_obj);
feat_freak->compute(fto_objeto, keypoints_surf_objeto, D_surf_freak_obj);
feat_brief->compute(fto_objeto, keypoints_surf_objeto, D_surf_brief_obj);
feat_orb->compute(fto_objeto, keypoints_surf_objeto, D_surf_orb_obj);

// Cálculo de descriptores para la ESCENA
feat_sift->compute(fto_scene, keypoints_surf_scene, D_surf_sift_scene);
feat_surf->compute(fto_scene, keypoints_surf_scene, D_surf_surf_scene);
feat_brisk->compute(fto_scene, keypoints_surf_scene, D_surf_brisk_scene);
feat_freak->compute(fto_scene, keypoints_surf_scene, D_surf_freak_scene);
feat_brief->compute(fto_scene, keypoints_surf_scene, D_surf_brief_scene);
feat_orb->compute(fto_scene, keypoints_surf_scene, D_surf_orb_scene);

//--- 20. SIFT-SIFT (L2) - SURF KP Base --------------------------------------
std::vector<DMatch> M_surf_sift;
Mat img_surf_sift;
FindGoodMatches(D_surf_sift_obj, D_surf_sift_scene, M_surf_sift, 0); // 0 = L2
cout << "20. SURF-SIFT Good Matches: " << M_surf_sift.size() << endl;
img_surf_sift = DrawHomography(fto_objeto, fto_scene, keypoints_surf_objeto, keypoints_surf_scene, M_surf_sift, "20. SURF-SIFT");

//--- 21. SURF-SURF (L2) - SURF KP Base --------------------------------------
std::vector<DMatch> M_surf_surf;
Mat img_surf_surf;
FindGoodMatches(D_surf_surf_obj, D_surf_surf_scene, M_surf_surf, 0); // 0 = L2
cout << "21. SURF-SURF Good Matches: " << M_surf_surf.size() << endl;
img_surf_surf = DrawHomography(fto_objeto, fto_scene, keypoints_surf_objeto, keypoints_surf_scene, M_surf_surf, "21. SURF-SURF");

//--- 22. BRISK-BRISK (Hamming) - SURF KP Base -------------------------------
std::vector<DMatch> M_surf_brisk;
Mat img_surf_brisk;
FindGoodMatches(D_surf_brisk_obj, D_surf_brisk_scene, M_surf_brisk, 1); // 1 = Hamming
cout << "22. SURF-BRISK Good Matches: " << M_surf_brisk.size() << endl;
img_surf_brisk = DrawHomography(fto_objeto, fto_scene, keypoints_surf_objeto, keypoints_surf_scene, M_surf_brisk, "22. SURF-BRISK");

//--- 23. FREAK-FREAK (Hamming) - SURF KP Base -------------------------------
std::vector<DMatch> M_surf_freak;
Mat img_surf_freak;
FindGoodMatches(D_surf_freak_obj, D_surf_freak_scene, M_surf_freak, 1); // 1 = Hamming
cout << "23. SURF-FREAK Good Matches: " << M_surf_freak.size() << endl;
img_surf_freak = DrawHomography(fto_objeto, fto_scene, keypoints_surf_objeto, keypoints_surf_scene, M_surf_freak, "23. SURF-FREAK");

//--- 24. BRIEF-BRIEF (Hamming) - SURF KP Base -------------------------------
std::vector<DMatch> M_surf_brief;
Mat img_surf_brief;
FindGoodMatches(D_surf_brief_obj, D_surf_brief_scene, M_surf_brief, 1); // 1 = Hamming
cout << "24. SURF-BRIEF Good Matches: " << M_surf_brief.size() << endl;
img_surf_brief = DrawHomography(fto_objeto, fto_scene, keypoints_surf_objeto, keypoints_surf_scene, M_surf_brief, "24. SURF-BRIEF");

//--- 25. ORB-ORB (Hamming) - SURF KP Base -----------------------------------
std::vector<DMatch> M_surf_orb;
Mat img_surf_orb;
FindGoodMatches(D_surf_orb_obj, D_surf_orb_scene, M_surf_orb, 1); // 1 = Hamming
cout << "25. SURF-ORB Good Matches: " << M_surf_orb.size() << endl;
img_surf_orb = DrawHomography(fto_objeto, fto_scene, keypoints_surf_objeto, keypoints_surf_scene, M_surf_orb, "25. SURF-ORB");
//--------------------------------------------------------------------------------------------------
//--------------- CÁLCULO DE DESCRIPTORES CON BASE SIFT KEYPOINTS (Para pruebas 13-19) -------------
//--------------- Se reutilizan los descriptores D_sift_* de la sección inicial --------------------

//--- 13. BRIEF-BRIEF (Hamming) - SIFT KP Base --------------------------------
// Se reutilizan D_sift_brief_obj y D_sift_brief_scene
std::vector<DMatch> M_brief_brief;
Mat img_brief_brief;
FindGoodMatches(D_sift_brief_obj, D_sift_brief_scene, M_brief_brief, 1); // 1 = Hamming
cout << "13. BRIEF-BRIEF Good Matches: " << M_brief_brief.size() << endl;
img_brief_brief = DrawHomography(fto_objeto, fto_scene, keypoints_sift_objeto, keypoints_sift_scene, M_brief_brief, "13. BRIEF-BRIEF");

//--- 14. ORB-ORB (Hamming) - SIFT KP Base ------------------------------------
// Se reutilizan D_sift_orb_obj y D_sift_orb_scene
std::vector<DMatch> M_brief_orb_sift_kp; // Renombrado para claridad
Mat img_brief_orb_sift_kp;
FindGoodMatches(D_sift_orb_obj, D_sift_orb_scene, M_brief_orb_sift_kp, 1); // 1 = Hamming
cout << "14. BRIEF-ORB Good Matches: " << M_brief_orb_sift_kp.size() << endl;
img_brief_orb_sift_kp = DrawHomography(fto_objeto, fto_scene, keypoints_sift_objeto, keypoints_sift_scene, M_brief_orb_sift_kp, "14. BRIEF-ORB (SIFT Detector)");

//--- 15. SURF-SURF (L2) - SIFT KP Base ---------------------------------------
// Se reutilizan D_sift_surf_obj y D_sift_surf_scene
std::vector<DMatch> M_brief_surf;
Mat img_brief_surf;
FindGoodMatches(D_sift_surf_obj, D_sift_surf_scene, M_brief_surf, 0); // 0 = L2
cout << "15. BRIEF-SURF Good Matches: " << M_brief_surf.size() << endl;
img_brief_surf = DrawHomography(fto_objeto, fto_scene, keypoints_sift_objeto, keypoints_sift_scene, M_brief_surf, "15. BRIEF-SURF");

//--- 16. FREAK-FREAK (Hamming) - SIFT KP Base --------------------------------
// Se reutilizan D_sift_freak_obj y D_sift_freak_scene
std::vector<DMatch> M_brief_freak;
Mat img_brief_freak;
FindGoodMatches(D_sift_freak_obj, D_sift_freak_scene, M_brief_freak, 1); // 1 = Hamming
cout << "16. BRIEF-FREAK Good Matches: " << M_brief_freak.size() << endl;
img_brief_freak = DrawHomography(fto_objeto, fto_scene, keypoints_sift_objeto, keypoints_sift_scene, M_brief_freak, "16. BRIEF-FREAK");

//--- 17. SIFT-SIFT (L2) - SIFT KP Base ---------------------------------------
// Se reutilizan D_sift_sift_obj y D_sift_sift_scene
std::vector<DMatch> M_brief_sift;
Mat img_brief_sift;
FindGoodMatches(D_sift_sift_obj, D_sift_sift_scene, M_brief_sift, 0); // 0 = L2
cout << "17. BRIEF-SIFT Good Matches: " << M_brief_sift.size() << endl;
img_brief_sift = DrawHomography(fto_objeto, fto_scene, keypoints_sift_objeto, keypoints_sift_scene, M_brief_sift, "17. BRIEF-SIFT");

//--- 18. BRISK-BRISK (Hamming) - SIFT KP Base --------------------------------
// Se reutilizan D_sift_brisk_obj y D_sift_brisk_scene
std::vector<DMatch> M_brief_brisk;
Mat img_brief_brisk;
FindGoodMatches(D_sift_brisk_obj, D_sift_brisk_scene, M_brief_brisk, 1); // 1 = Hamming
cout << "18. BRIEF-BRISK Good Matches: " << M_brief_brisk.size() << endl;
img_brief_brisk = DrawHomography(fto_objeto, fto_scene, keypoints_sift_objeto, keypoints_sift_scene, M_brief_brisk, "18. BRIEF-BRISK");

//--- 19. ORB-ORB (Hamming) - SIFT KP Base ------------------------------------
// Se reutilizan D_sift_orb_obj y D_sift_orb_scene
std::vector<DMatch> M_brief_orb_2; 
Mat img_brief_orb_2;
FindGoodMatches(D_sift_orb_obj, D_sift_orb_scene, M_brief_orb_2, 1); // 1 = Hamming
cout << "19. BRIEF-ORB Good Matches: " << M_brief_orb_2.size() << endl;
img_brief_orb_2 = DrawHomography(fto_objeto, fto_scene, keypoints_sift_objeto, keypoints_sift_scene, M_brief_orb_2, "19. BRIEF-ORB (Descriptor)");


//---------------VISUALIZACIONES (Solo si la imagen NO está vacía) -----------------------------------

    if (!img_sift_sift.empty()) imshow("1. SIFT-SIFT", img_sift_sift);
    if (!img_sift_surf.empty()) imshow("2. SIFT-SURF", img_sift_surf);
    if (!img_sift_brisk.empty()) imshow("3. SIFT-BRISK", img_sift_brisk);
    if (!img_sift_freak.empty()) imshow("4. SIFT-FREAK", img_sift_freak);
    if (!img_sift_brief.empty()) imshow("5. SIFT-BRIEF", img_sift_brief);
    if (!img_sift_orb.empty()) imshow("6. SIFT-ORB", img_sift_orb);

    if (!img_fast_sift.empty()) imshow("7. FAST-SIFT", img_fast_sift);
    if (!img_fast_brisk.empty()) imshow("8. FAST-BRISK", img_fast_brisk);
    if (!img_fast_surf.empty()) imshow("9. FAST-SURF", img_fast_surf);
    if (!img_fast_orb.empty()) imshow("10. FAST-ORB", img_fast_orb);
    if (!img_fast_brief.empty()) imshow("11. FAST-BRIEF", img_fast_brief);
    if (!img_fast_freak.empty()) imshow("12. FAST-FREAK", img_fast_freak);
    
    if (!img_surf_sift.empty()) imshow("20. SURF-SIFT", img_surf_sift);
    if (!img_surf_surf.empty()) imshow("21. SURF-SURF", img_surf_surf);
    if (!img_surf_brisk.empty()) imshow("22. SURF-BRISK", img_surf_brisk);
    if (!img_surf_freak.empty()) imshow("23. SURF-FREAK", img_surf_freak);
    if (!img_surf_brief.empty()) imshow("24. SURF-BRIEF", img_surf_brief);
    if (!img_surf_orb.empty()) imshow("25. SURF-ORB", img_surf_orb);
    
    if (!img_brief_brief.empty()) imshow("13. BRIEF-BRIEF", img_brief_brief);
    if (!img_brief_orb_sift_kp.empty()) imshow("14. BRIEF-ORB (SIFT Base)", img_brief_orb_sift_kp);
    if (!img_brief_surf.empty()) imshow("15. BRIEF-SURF", img_brief_surf);
    if (!img_brief_freak.empty()) imshow("16. BRIEF-FREAK", img_brief_freak);
    if (!img_brief_sift.empty()) imshow("17. BRIEF-SIFT", img_brief_sift);
    if (!img_brief_brisk.empty()) imshow("18. BRIEF-BRISK", img_brief_brisk);
    if (!img_brief_orb_2.empty()) imshow("19. BRIEF-ORB (ORB Descriptor)", img_brief_orb_2);


    // Solo llamar a waitKey si se mostró al menos una imagen
    if (!img_sift_sift.empty() || !img_sift_surf.empty() || !img_sift_brisk.empty() || 
        !img_sift_freak.empty() || !img_sift_brief.empty() || !img_sift_orb.empty() ||
        !img_fast_sift.empty() || !img_fast_brisk.empty() || !img_fast_surf.empty() || 
        !img_fast_orb.empty() || !img_fast_brief.empty() || !img_fast_freak.empty() ||
        !img_surf_sift.empty() || !img_surf_surf.empty() || !img_surf_brisk.empty() || 
        !img_surf_freak.empty() || !img_surf_brief.empty() || !img_surf_orb.empty() ||
        !img_brief_brief.empty() || !img_brief_orb_sift_kp.empty() || !img_brief_surf.empty() || 
        !img_brief_freak.empty() || !img_brief_sift.empty() || !img_brief_brisk.empty() || !img_brief_orb_2.empty()) {
        waitKey();
    }
    
    return 0;
}