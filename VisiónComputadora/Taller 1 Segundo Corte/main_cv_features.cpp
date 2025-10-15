#include <stdint.h>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

// ----------------------
// Librerías de OpenCV
// ----------------------
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/tracking.hpp"
#include "opencv2/videoio.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

// -------------------------------------------------------
// Función: contarMatches
// -------------------------------------------------------
// Esta función detecta keypoints y calcula descriptores en
// dos imágenes usando los algoritmos que se pasan como parámetros.
// Luego, compara ambos conjuntos de descriptores con un
// emparejador (matcher) y cuenta los "matches buenos".
// -------------------------------------------------------
int contarMatches(const Ptr<Feature2D>& detector,     // Algoritmo para detectar keypoints
    const Ptr<Feature2D>& descriptor,   // Algoritmo para calcular descriptores
    const Mat& img1,                    // Imagen 1 (objeto de referencia)
    const Mat& img2,                    // Imagen 2 (escena completa)
    bool binario) {                     // True si el descriptor usa NORM_HAMMING (binario), false si usa NORM_L2 (float)

    // 1️⃣ Detectar keypoints en ambas imágenes
    vector<KeyPoint> kp1, kp2;
    detector->detect(img1, kp1);
    detector->detect(img2, kp2);

    if (kp1.empty() || kp2.empty()) {
        cerr << "  ⚠️ Sin keypoints detectados." << endl;
        return 0;
    }

    // 2️⃣ Calcular descriptores
    Mat desc1, desc2;
    descriptor->compute(img1, kp1, desc1);
    descriptor->compute(img2, kp2, desc2);

    if (desc1.empty() || desc2.empty()) {
        cerr << "  ⚠️ No se generaron descriptores." << endl;
        return 0;
    }

    // 3️⃣ Crear el matcher adecuado según el tipo de descriptor
    //    - NORM_HAMMING → descriptores binarios (ORB, BRISK, BRIEF, FREAK)
    //    - NORM_L2 → descriptores flotantes (SIFT, SURF)
    BFMatcher matcher(binario ? NORM_HAMMING : NORM_L2);
    vector<DMatch> matches;
    matcher.match(desc1, desc2, matches);

    // 4️⃣ Encontrar la menor distancia de todos los matches
    double min_dist = 100;
    for (auto& m : matches)
        min_dist = min(min_dist, (double)m.distance);

    // 5️⃣ Filtrar los "buenos matches"
    //     Se considera bueno si la distancia es menor que 2 × min_dist
    vector<DMatch> good_matches;
    for (auto& m : matches)
        if (m.distance <= max(2 * min_dist, 0.02))
            good_matches.push_back(m);

    // 6️⃣ Devolver cuántos matches buenos se encontraron
    return (int)good_matches.size();
}

// -------------------------------------------------------
// Función principal (main)
// -------------------------------------------------------
int main() {
    // Rutas de las imágenes a comparar
    string nom_objeto = "C:/Users/Juan/Documents/Repositorio/VisiónComputadora/Img_Corte2/box.png";
    string nom_scene = "C:/Users/Juan/Documents/Repositorio/VisiónComputadora/Img_Corte2/box_in_scene.png";

    // Cargar imágenes en escala de grises (necesario para detectores de características)
    Mat img1 = imread(nom_objeto, IMREAD_GRAYSCALE);
    Mat img2 = imread(nom_scene, IMREAD_GRAYSCALE);

    // Verificar que las imágenes se hayan cargado correctamente
    if (img1.empty() || img2.empty()) {
        cerr << "Error al cargar las imágenes." << endl;
        return -1;
    }

    // Encabezado informativo
    cout << "=============================================\n";
    cout << "   Comparación de detectores y descriptores  \n";
    cout << "   (Keypoint: BRISK, ORB, FREAK | Modelo: Fuerza Bruta)\n";
    cout << "=============================================\n\n";

    // -------------------------------------------------------
    // Estructura para definir cada combinación a probar
    // -------------------------------------------------------
    struct Combinacion {
        string nombre;              // Descripción de la combinación
        bool posible;               // Indica si se puede ejecutar o no
        Ptr<Feature2D> detector;    // Algoritmo de detección
        Ptr<Feature2D> descriptor;  // Algoritmo de descripción
        bool binario;               // Si usa NORM_HAMMING (true) o NORM_L2 (false)
    };

    // -------------------------------------------------------
    // Lista de combinaciones posibles entre detectores y descriptores
    // -------------------------------------------------------
    vector<Combinacion> combinaciones = {
        {"BRISK + BRISK", true, BRISK::create(), BRISK::create(), true},
        {"BRISK + SIFT",  true, BRISK::create(), SIFT::create(),  false},
        {"BRISK + ORB",   true, BRISK::create(), ORB::create(),   true},
        {"BRISK + BRIEF", true, BRISK::create(), BriefDescriptorExtractor::create(), true},
        {"BRISK + FREAK", true, BRISK::create(), FREAK::create(), true},
        {"ORB + BRISK",   true, ORB::create(),   BRISK::create(), true},
        {"ORB + SIFT",    true, ORB::create(),   SIFT::create(),  false},
        {"ORB + ORB",     true, ORB::create(),   ORB::create(),   true},
        {"ORB + BRIEF",   true, ORB::create(),   BriefDescriptorExtractor::create(), true},
        {"ORB + FREAK",   true, ORB::create(),   FREAK::create(), true},

        // FREAK no puede detectar keypoints (solo describir), por eso son "no posibles"
        {"FREAK + BRISK", false, nullptr, nullptr, true},
        {"FREAK + SIFT",  false, nullptr, nullptr, false},
        {"FREAK + ORB",   false, nullptr, nullptr, true},
        {"FREAK + BRIEF", false, nullptr, nullptr, true},
        {"FREAK + FREAK", false, nullptr, nullptr, true}
    };

    // -------------------------------------------------------
    // Bucle principal: probar cada combinación
    // -------------------------------------------------------
    for (auto& comb : combinaciones) {
        cout << "▶ " << comb.nombre << endl;

        // Caso especial: FREAK no es detector → no se puede usar como tal
        if (!comb.posible) {
            cout << "   ❌ Combinación no posible (FREAK no es detector de keypoints)\n\n";
            continue;
        }

        // Ejecutar el conteo de matches y mostrar resultado
        int n_matches = contarMatches(comb.detector, comb.descriptor, img1, img2, comb.binario);
        cout << "   → Matches válidos: " << n_matches << "\n\n";
    }

    // -------------------------------------------------------
    // Fin del programa
    // -------------------------------------------------------
    cout << "=============================================\n";
    cout << "Comparación terminada.\n";
    cout << "=============================================\n";

    return 0;
}