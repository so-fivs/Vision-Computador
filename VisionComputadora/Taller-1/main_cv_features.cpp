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
        // Hamming para descriptores binarios
        matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
    }
    else 
    {
        // L2 para descriptores flotantes
        matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    }
    
    std::vector< std::vector<DMatch> > matches_knn;
    const float ratio_thresh = 0.75f; 

    // Try-catch dentro de FindGoodMatches
    try {
        matcher->knnMatch(descriptors_object, descriptors_scene, matches_knn, 2);
    } catch (const cv::Exception& e) {
        // Falló la comparación (p. ej., tipos de datos incompatibles o problema de tamaño)
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

    // ***************************************************************
    // VERIFICACIÓN CLAVE: Si 2 o menos matches, se considera no compatible y retorna Mat vacía.
    // ***************************************************************
    if (num_matches <= 2) 
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

    if (obj_points.size() <= 2)
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
            
            line(img_matches, scene_corners[0] + Point2f((float)fto_objeto.cols, 0), 
                 scene_corners[1] + Point2f((float)fto_objeto.cols, 0), Scalar(0, 255, 0), 4);
            line(img_matches, scene_corners[1] + Point2f((float)fto_objeto.cols, 0), 
                 scene_corners[2] + Point2f((float)fto_objeto.cols, 0), Scalar(0, 255, 0), 4);
            line(img_matches, scene_corners[2] + Point2f((float)fto_objeto.cols, 0), 
                 scene_corners[3] + Point2f((float)fto_objeto.cols, 0), Scalar(0, 255, 0), 4);
            line(img_matches, scene_corners[3] + Point2f((float)fto_objeto.cols, 0), 
                 scene_corners[0] + Point2f((float)fto_objeto.cols, 0), Scalar(0, 255, 0), 4);
            
        } catch (const cv::Exception& e) {
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
   Ptr<SURF> feat_surf = SURF::create(500, 4, 4, false, false); 
   Ptr<BRISK> feat_brisk = BRISK::create(500, 20, 1.0f);
   Ptr<ORB> feat_orb = ORB::create(500, 1.2f, 12, 31, 0, 2, ORB::HARRIS_SCORE, 20, 20); 
   Ptr<FREAK> feat_freak = FREAK::create(true, true, 32.0f, 10);
   Ptr<xfeatures2d::BriefDescriptorExtractor> feat_brief = xfeatures2d::BriefDescriptorExtractor::create(16, true);
   Ptr<FastFeatureDetector> detector_fast = FastFeatureDetector::create(10, false, FastFeatureDetector::TYPE_9_16);


//---------------DETECCIÓN DE KEYPOINTS SIFT----------------------------------

   std::vector<KeyPoint> keypoints_sift_objeto_BASE, keypoints_sift_scene_BASE;
   feat_sift->detect(fto_objeto, keypoints_sift_objeto_BASE);
   feat_sift->detect(fto_scene, keypoints_sift_scene_BASE);
   
   cout << "SIFT KeyPoints Objeto: " << keypoints_sift_objeto_BASE.size() << endl;
   cout << "SIFT KeyPoints Escena: " << keypoints_sift_scene_BASE.size() << endl;

//--- 1. SIFT - SIFT (L2) ----------------------------------------------------
   Mat D_sift_sift_obj, D_sift_sift_scene;
   std::vector<DMatch> M_sift_sift;
   Mat img_sift_sift;
   try {
       feat_sift->compute(fto_objeto, keypoints_sift_objeto_BASE, D_sift_sift_obj);
       feat_sift->compute(fto_scene, keypoints_sift_scene_BASE, D_sift_sift_scene);
       FindGoodMatches(D_sift_sift_obj, D_sift_sift_scene, M_sift_sift, 0);
       cout << "SIFT-SIFT Good Matches: " << M_sift_sift.size() << endl;
       img_sift_sift = DrawHomography(fto_objeto, fto_scene, keypoints_sift_objeto_BASE, keypoints_sift_scene_BASE, M_sift_sift, "1. SIFT-SIFT");
   } catch (const cv::Exception& e) {
       cout << "ERROR en 1. SIFT-SIFT: " << e.what() << endl;
   }

//--- 2. SIFT - SURF (L2) ----------------------------------------------------
   Mat D_sift_surf_obj, D_sift_surf_scene;
   std::vector<DMatch> M_sift_surf;
   Mat img_sift_surf;
   try {
       feat_surf->compute(fto_objeto, keypoints_sift_objeto_BASE, D_sift_surf_obj);
       feat_surf->compute(fto_scene, keypoints_sift_scene_BASE, D_sift_surf_scene);
       FindGoodMatches(D_sift_surf_obj, D_sift_surf_scene, M_sift_surf, 0);
       cout << "SIFT-SURF Good Matches: " << M_sift_surf.size() << endl;
       img_sift_surf = DrawHomography(fto_objeto, fto_scene, keypoints_sift_objeto_BASE, keypoints_sift_scene_BASE, M_sift_surf, "2. SIFT-SURF");
   } catch (const cv::Exception& e) {
       cout << "ERROR en 2. SIFT-SURF: " << e.what() << endl;
   }

//--- 3. SIFT - BRISK (Hamming) ----------------------------------------------
   Mat D_sift_brisk_obj, D_sift_brisk_scene;
   std::vector<DMatch> M_sift_brisk;
   Mat img_sift_brisk;
   try {
       feat_brisk->compute(fto_objeto, keypoints_sift_objeto_BASE, D_sift_brisk_obj);
       feat_brisk->compute(fto_scene, keypoints_sift_scene_BASE, D_sift_brisk_scene);
       FindGoodMatches(D_sift_brisk_obj, D_sift_brisk_scene, M_sift_brisk, 1);
       cout << "SIFT-BRISK Good Matches: " << M_sift_brisk.size() << endl;
       img_sift_brisk = DrawHomography(fto_objeto, fto_scene, keypoints_sift_objeto_BASE, keypoints_sift_scene_BASE, M_sift_brisk, "3. SIFT-BRISK");
   } catch (const cv::Exception& e) {
       cout << "ERROR en 3. SIFT-BRISK: " << e.what() << endl;
   }

//--- 4. SIFT - FREAK (Hamming) ----------------------------------------------
   Mat D_sift_freak_obj, D_sift_freak_scene;
   std::vector<DMatch> M_sift_freak;
   Mat img_sift_freak;
   try {
       feat_freak->compute(fto_objeto, keypoints_sift_objeto_BASE, D_sift_freak_obj);
       feat_freak->compute(fto_scene, keypoints_sift_scene_BASE, D_sift_freak_scene);
       FindGoodMatches(D_sift_freak_obj, D_sift_freak_scene, M_sift_freak, 1);
       cout << "SIFT-FREAK Good Matches: " << M_sift_freak.size() << endl;
       img_sift_freak = DrawHomography(fto_objeto, fto_scene, keypoints_sift_objeto_BASE, keypoints_sift_scene_BASE, M_sift_freak, "4. SIFT-FREAK");
   } catch (const cv::Exception& e) {
       cout << "ERROR en 4. SIFT-FREAK: " << e.what() << endl;
   }

//--- 5. SIFT - BRIEF (Hamming) ----------------------------------------------
   Mat D_sift_brief_obj, D_sift_brief_scene;
   std::vector<DMatch> M_sift_brief;
   Mat img_sift_brief;
   try {
       feat_brief->compute(fto_objeto, keypoints_sift_objeto_BASE, D_sift_brief_obj);
       feat_brief->compute(fto_scene, keypoints_sift_scene_BASE, D_sift_brief_scene);
       FindGoodMatches(D_sift_brief_obj, D_sift_brief_scene, M_sift_brief, 1);
       cout << "SIFT-BRIEF Good Matches: " << M_sift_brief.size() << endl;
       img_sift_brief = DrawHomography(fto_objeto, fto_scene, keypoints_sift_objeto_BASE, keypoints_sift_scene_BASE, M_sift_brief, "5. SIFT-BRIEF");
   } catch (const cv::Exception& e) {
       cout << "ERROR en 5. SIFT-BRIEF: " << e.what() << endl;
   }

//--- 6. SIFT - ORB (Hamming) ------------------------------------------------
   Mat D_sift_orb_obj, D_sift_orb_scene;
   std::vector<DMatch> M_sift_orb;
   Mat img_sift_orb;
   try {
       feat_orb->compute(fto_objeto, keypoints_sift_objeto_BASE, D_sift_orb_obj);
       feat_orb->compute(fto_scene, keypoints_sift_scene_BASE, D_sift_orb_scene);
       FindGoodMatches(D_sift_orb_obj, D_sift_orb_scene, M_sift_orb, 1);
       cout << "SIFT-ORB Good Matches: " << M_sift_orb.size() << endl;
       img_sift_orb = DrawHomography(fto_objeto, fto_scene, keypoints_sift_objeto_BASE, keypoints_sift_scene_BASE, M_sift_orb, "6. SIFT-ORB");
   } catch (const cv::Exception& e) {
       cout << "ERROR en 6. SIFT-ORB: " << e.what() << endl;
   }
    
//--------------------------------------------------------------------------------------------------
//---------------DETECCIÓN DE KEYPOINTS FAST -----------------------------

   std::vector<KeyPoint> keypoints_fast_objeto_BASE, keypoints_fast_scene_BASE;
   detector_fast->detect(fto_objeto, keypoints_fast_objeto_BASE);
   detector_fast->detect(fto_scene, keypoints_fast_scene_BASE);
   
   cout << "FAST KeyPoints Objeto: " << keypoints_fast_objeto_BASE.size() << endl;
   cout << "FAST KeyPoints Escena: " << keypoints_fast_scene_BASE.size() << endl;


//--- 7. FAST - SIFT (L2) ----------------------------------------------------
   Mat D_fast_sift_obj, D_fast_sift_scene;
   std::vector<DMatch> M_fast_sift;
   Mat img_fast_sift;
   try {
       feat_sift->compute(fto_objeto, keypoints_fast_objeto_BASE, D_fast_sift_obj);
       feat_sift->compute(fto_scene, keypoints_fast_scene_BASE, D_fast_sift_scene);
       FindGoodMatches(D_fast_sift_obj, D_fast_sift_scene, M_fast_sift, 0);
       cout << "FAST-SIFT Good Matches: " << M_fast_sift.size() << endl;
       img_fast_sift = DrawHomography(fto_objeto, fto_scene, keypoints_fast_objeto_BASE, keypoints_fast_scene_BASE, M_fast_sift, "7. FAST-SIFT");
   } catch (const cv::Exception& e) {
       cout << "ERROR en 7. FAST-SIFT: " << e.what() << endl;
   }


//--- 8. FAST - BRISK (Hamming) ----------------------------------------------
   Mat D_fast_brisk_obj, D_fast_brisk_scene;
   std::vector<DMatch> M_fast_brisk;
   Mat img_fast_brisk;
   try {
       feat_brisk->compute(fto_objeto, keypoints_fast_objeto_BASE, D_fast_brisk_obj);
       feat_brisk->compute(fto_scene, keypoints_fast_scene_BASE, D_fast_brisk_scene);
       FindGoodMatches(D_fast_brisk_obj, D_fast_brisk_scene, M_fast_brisk, 1);
       cout << "FAST-BRISK Good Matches: " << M_fast_brisk.size() << endl;
       img_fast_brisk = DrawHomography(fto_objeto, fto_scene, keypoints_fast_objeto_BASE, keypoints_fast_scene_BASE, M_fast_brisk, "8. FAST-BRISK");
   } catch (const cv::Exception& e) {
       cout << "ERROR en 8. FAST-BRISK: " << e.what() << endl;
   }


//--- 9. FAST - SURF (L2) ----------------------------------------------------
   Mat D_fast_surf_obj, D_fast_surf_scene;
   std::vector<DMatch> M_fast_surf;
   Mat img_fast_surf;
   try {
       feat_surf->compute(fto_objeto, keypoints_fast_objeto_BASE, D_fast_surf_obj);
       feat_surf->compute(fto_scene, keypoints_fast_scene_BASE, D_fast_surf_scene);
       FindGoodMatches(D_fast_surf_obj, D_fast_surf_scene, M_fast_surf, 0);
       cout << "FAST-SURF Good Matches: " << M_fast_surf.size() << endl;
       img_fast_surf = DrawHomography(fto_objeto, fto_scene, keypoints_fast_objeto_BASE, keypoints_fast_scene_BASE, M_fast_surf, "9. FAST-SURF");
   } catch (const cv::Exception& e) {
       cout << "ERROR en 9. FAST-SURF: " << e.what() << endl;
   }


//--- 10. FAST - ORB (Hamming) -----------------------------------------------
   Mat D_fast_orb_obj, D_fast_orb_scene;
   std::vector<DMatch> M_fast_orb;
   Mat img_fast_orb;
   try {
       feat_orb->compute(fto_objeto, keypoints_fast_objeto_BASE, D_fast_orb_obj);
       feat_orb->compute(fto_scene, keypoints_fast_scene_BASE, D_fast_orb_scene);
       FindGoodMatches(D_fast_orb_obj, D_fast_orb_scene, M_fast_orb, 1);
       cout << "FAST-ORB Good Matches: " << M_fast_orb.size() << endl;
       img_fast_orb = DrawHomography(fto_objeto, fto_scene, keypoints_fast_objeto_BASE, keypoints_fast_scene_BASE, M_fast_orb, "10. FAST-ORB");
   } catch (const cv::Exception& e) {
       cout << "ERROR en 10. FAST-ORB: " << e.what() << endl;
   }


//--- 11. FAST - BRIEF (Hamming) ---------------------------------------------
   Mat D_fast_brief_obj, D_fast_brief_scene;
   std::vector<DMatch> M_fast_brief;
   Mat img_fast_brief;
   try {
       feat_brief->compute(fto_objeto, keypoints_fast_objeto_BASE, D_fast_brief_obj);
       feat_brief->compute(fto_scene, keypoints_fast_scene_BASE, D_fast_brief_scene);
       FindGoodMatches(D_fast_brief_obj, D_fast_brief_scene, M_fast_brief, 1);
       cout << "FAST-BRIEF Good Matches: " << M_fast_brief.size() << endl;
       img_fast_brief = DrawHomography(fto_objeto, fto_scene, keypoints_fast_objeto_BASE, keypoints_fast_scene_BASE, M_fast_brief, "11. FAST-BRIEF");
   } catch (const cv::Exception& e) {
       cout << "ERROR en 11. FAST-BRIEF: " << e.what() << endl;
   }


//--- 12. FAST - FREAK (Hamming) ---------------------------------------------
   Mat D_fast_freak_obj, D_fast_freak_scene;
   std::vector<DMatch> M_fast_freak;
   Mat img_fast_freak;
   try {
       feat_freak->compute(fto_objeto, keypoints_fast_objeto_BASE, D_fast_freak_obj);
       feat_freak->compute(fto_scene, keypoints_fast_scene_BASE, D_fast_freak_scene);
       FindGoodMatches(D_fast_freak_obj, D_fast_freak_scene, M_fast_freak, 1);
       cout << "FAST-FREAK Good Matches: " << M_fast_freak.size() << endl;
       img_fast_freak = DrawHomography(fto_objeto, fto_scene, keypoints_fast_objeto_BASE, keypoints_fast_scene_BASE, M_fast_freak, "12. FAST-FREAK");
   } catch (const cv::Exception& e) {
       cout << "ERROR en 12. FAST-FREAK: " << e.what() << endl;
   }

//--------------------------------------------------------------------------------------------------
//---------------DETECCIÓN DE KEYPOINTS BRIEF (Usando FAST como detector base)----------------------

   // En esta sección se utilizan los KeyPoints generados por FAST (keypoints_fast_objeto_BASE/scene_BASE) 
   // ya que BRIEF es solo un descriptor, no un detector.

//--- 13. BRIEF - BRIEF (Hamming) --------------------------------------------
   Mat D_brief_brief_obj, D_brief_brief_scene;
   std::vector<DMatch> M_brief_brief;
   Mat img_brief_brief;
   try {
       feat_brief->compute(fto_objeto, keypoints_fast_objeto_BASE, D_brief_brief_obj);
       feat_brief->compute(fto_scene, keypoints_fast_scene_BASE, D_brief_brief_scene);
       FindGoodMatches(D_brief_brief_obj, D_brief_brief_scene, M_brief_brief, 1);
       cout << "BRIEF-BRIEF Good Matches: " << M_brief_brief.size() << endl;
       img_brief_brief = DrawHomography(fto_objeto, fto_scene, keypoints_fast_objeto_BASE, keypoints_fast_scene_BASE, M_brief_brief, "13. BRIEF-BRIEF");
   } catch (const cv::Exception& e) {
       cout << "ERROR en 13. BRIEF-BRIEF: " << e.what() << endl;
   }

//--- 14. BRIEF - FAST (Hamming) ---------------------------------------------
   Mat D_brief_orb_obj, D_brief_orb_scene; 
   std::vector<DMatch> M_brief_orb;
   Mat img_brief_orb;
   try {
       feat_orb->compute(fto_objeto, keypoints_fast_objeto_BASE, D_brief_orb_obj);
       feat_orb->compute(fto_scene, keypoints_fast_scene_BASE, D_brief_orb_scene);
       FindGoodMatches(D_brief_orb_obj, D_brief_orb_scene, M_brief_orb, 1);
       cout << "BRIEF-ORB Good Matches: " << M_brief_orb.size() << endl;
       img_brief_orb = DrawHomography(fto_objeto, fto_scene, keypoints_fast_objeto_BASE, keypoints_fast_scene_BASE, M_brief_orb, "14. BRIEF-ORB (FAST Detector)");
   } catch (const cv::Exception& e) {
       cout << "ERROR en 14. BRIEF-ORB: " << e.what() << endl;
   }


//--- 15. BRIEF - SURF (L2) --------------------------------------------------
   Mat D_brief_surf_obj, D_brief_surf_scene;
   std::vector<DMatch> M_brief_surf;
   Mat img_brief_surf;
   try {
       feat_surf->compute(fto_objeto, keypoints_fast_objeto_BASE, D_brief_surf_obj);
       feat_surf->compute(fto_scene, keypoints_fast_scene_BASE, D_brief_surf_scene);
       FindGoodMatches(D_brief_surf_obj, D_brief_surf_scene, M_brief_surf, 0); // L2
       cout << "BRIEF-SURF Good Matches: " << M_brief_surf.size() << endl;
       img_brief_surf = DrawHomography(fto_objeto, fto_scene, keypoints_fast_objeto_BASE, keypoints_fast_scene_BASE, M_brief_surf, "15. BRIEF-SURF");
   } catch (const cv::Exception& e) {
       cout << "ERROR en 15. BRIEF-SURF: " << e.what() << endl;
   }

//--- 16. BRIEF - FREAK (Hamming) --------------------------------------------
   Mat D_brief_freak_obj, D_brief_freak_scene;
   std::vector<DMatch> M_brief_freak;
   Mat img_brief_freak;
   try {
       feat_freak->compute(fto_objeto, keypoints_fast_objeto_BASE, D_brief_freak_obj);
       feat_freak->compute(fto_scene, keypoints_fast_scene_BASE, D_brief_freak_scene);
       FindGoodMatches(D_brief_freak_obj, D_brief_freak_scene, M_brief_freak, 1);
       cout << "BRIEF-FREAK Good Matches: " << M_brief_freak.size() << endl;
       img_brief_freak = DrawHomography(fto_objeto, fto_scene, keypoints_fast_objeto_BASE, keypoints_fast_scene_BASE, M_brief_freak, "16. BRIEF-FREAK");
   } catch (const cv::Exception& e) {
       cout << "ERROR en 16. BRIEF-FREAK: " << e.what() << endl;
   }

//--- 17. BRIEF - SIFT (L2) --------------------------------------------------
   Mat D_brief_sift_obj, D_brief_sift_scene;
   std::vector<DMatch> M_brief_sift;
   Mat img_brief_sift;
   try {
       feat_sift->compute(fto_objeto, keypoints_fast_objeto_BASE, D_brief_sift_obj);
       feat_sift->compute(fto_scene, keypoints_fast_scene_BASE, D_brief_sift_scene);
       FindGoodMatches(D_brief_sift_obj, D_brief_sift_scene, M_brief_sift, 0); // L2
       cout << "BRIEF-SIFT Good Matches: " << M_brief_sift.size() << endl;
       img_brief_sift = DrawHomography(fto_objeto, fto_scene, keypoints_fast_objeto_BASE, keypoints_fast_scene_BASE, M_brief_sift, "17. BRIEF-SIFT");
   } catch (const cv::Exception& e) {
       cout << "ERROR en 17. BRIEF-SIFT: " << e.what() << endl;
   }

//--- 18. BRIEF - BRISK (Hamming) --------------------------------------------
   Mat D_brief_brisk_obj, D_brief_brisk_scene;
   std::vector<DMatch> M_brief_brisk;
   Mat img_brief_brisk;
   try {
       feat_brisk->compute(fto_objeto, keypoints_fast_objeto_BASE, D_brief_brisk_obj);
       feat_brisk->compute(fto_scene, keypoints_fast_scene_BASE, D_brief_brisk_scene);
       FindGoodMatches(D_brief_brisk_obj, D_brief_brisk_scene, M_brief_brisk, 1);
       cout << "BRIEF-BRISK Good Matches: " << M_brief_brisk.size() << endl;
       img_brief_brisk = DrawHomography(fto_objeto, fto_scene, keypoints_fast_objeto_BASE, keypoints_fast_scene_BASE, M_brief_brisk, "18. BRIEF-BRISK");
   } catch (const cv::Exception& e) {
       cout << "ERROR en 18. BRIEF-BRISK: " << e.what() << endl;
   }

//--- 19. BRIEF - ORB (Hamming) ----------------------------------------------
   Mat D_brief_orb_2_obj, D_brief_orb_2_scene; // Se usa un nuevo nombre para evitar conflicto con la 14
   std::vector<DMatch> M_brief_orb_2;
   Mat img_brief_orb_2;
   try {
       feat_orb->compute(fto_objeto, keypoints_fast_objeto_BASE, D_brief_orb_2_obj);
       feat_orb->compute(fto_scene, keypoints_fast_scene_BASE, D_brief_orb_2_scene);
       FindGoodMatches(D_brief_orb_2_obj, D_brief_orb_2_scene, M_brief_orb_2, 1);
       cout << "BRIEF-ORB Good Matches: " << M_brief_orb_2.size() << endl;
       img_brief_orb_2 = DrawHomography(fto_objeto, fto_scene, keypoints_fast_objeto_BASE, keypoints_fast_scene_BASE, M_brief_orb_2, "19. BRIEF-ORB (Descriptor)");
   } catch (const cv::Exception& e) {
       cout << "ERROR en 19. BRIEF-ORB: " << e.what() << endl;
   }


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
    
    if (!img_brief_brief.empty()) imshow("13. BRIEF-BRIEF", img_brief_brief);
    if (!img_brief_orb.empty()) imshow("14. BRIEF-ORB (FAST Base)", img_brief_orb);
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
        !img_brief_brief.empty() || !img_brief_orb.empty() || !img_brief_surf.empty() || 
        !img_brief_freak.empty() || !img_brief_sift.empty() || !img_brief_brisk.empty() || !img_brief_orb_2.empty()) {
        waitKey();
    }
    
    return 0;
}