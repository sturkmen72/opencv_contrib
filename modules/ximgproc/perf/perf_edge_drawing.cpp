// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test
{
namespace
{
/* 1. Define parameter type and test fixture */
typedef tuple<int, int> EDTestParam;
typedef TestBaseWithParam<EDTestParam> EdgeDrawing_Detect_Edges;
typedef TestBaseWithParam<EDTestParam> EdgeDrawing_Detect_Lines;
typedef TestBaseWithParam<EDTestParam> EdgeDrawing_Detect_Ellipses;

/* 2. Declare the testsuite */
PERF_TEST_P(EdgeDrawing_Detect_Edges, perf,
            Combine(Values(0, 1, 2, 3), Values(10, 50, 150, 300)))
{
    /* 3. Get actual test parameters */
    EDTestParam params = GetParam();
    int EdgeDetectionOperator = get<0>(params);
    int GradientThresholdValue = get<1>(params);

    /* 4. Allocate and initialize arguments for tested function */
    std::string filename = getDataPath("perf/1680x1050.png");
    Mat src = imread(filename, 0);

    /* 5. Manifest your expectations about this test */
    declare.in(src);

    /* 6. Collect the samples! */
    PERF_SAMPLE_BEGIN();
    Ptr<ximgproc::EdgeDrawing> ed = ximgproc::createEdgeDrawing();
    ed->params.EdgeDetectionOperator = EdgeDetectionOperator;
    ed->params.GradientThresholdValue = GradientThresholdValue;
    ed->detectEdges(src);
    PERF_SAMPLE_END();

    /* 7. Do not check anything */
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(EdgeDrawing_Detect_Lines, perf,
            Combine(Values(0, 1, 2, 3), Values(10, 50, 150, 300)))
{
    EDTestParam params = GetParam();
    int EdgeDetectionOperator = get<0>(params);
    int GradientThresholdValue = get<1>(params);

    std::string filename = getDataPath("perf/1680x1050.png");
    Mat src = imread(filename, IMREAD_GRAYSCALE);

    declare.in(src);

    PERF_SAMPLE_BEGIN();
    Ptr<ximgproc::EdgeDrawing> ed = ximgproc::createEdgeDrawing();
    ed->params.EdgeDetectionOperator = EdgeDetectionOperator;
    ed->params.GradientThresholdValue = GradientThresholdValue;
    ed->detectEdges(src);
    std::vector<Vec4f> lines;
    ed->detectLines(lines);
    PERF_SAMPLE_END();

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(EdgeDrawing_Detect_Ellipses, perf,
            Combine(Values(0, 1, 2, 3), Values(10, 50, 150, 300)))
{
    EDTestParam params = GetParam();
    int EdgeDetectionOperator = get<0>(params);
    int GradientThresholdValue = get<1>(params);

    std::string filename = getDataPath("perf/1680x1050.png");
    Mat src = imread(filename, 0);

    declare.in(src);

    PERF_SAMPLE_BEGIN();
    Ptr<ximgproc::EdgeDrawing> ed = ximgproc::createEdgeDrawing();
    ed->params.EdgeDetectionOperator = EdgeDetectionOperator;
    ed->params.GradientThresholdValue = GradientThresholdValue;
    ed->detectEdges(src);
    vector<Vec6d> ellipses;
    ed->detectEllipses(ellipses);
    PERF_SAMPLE_END();

    SANITY_CHECK_NOTHING();
}
} // namespace
} // namespace opencv_test
