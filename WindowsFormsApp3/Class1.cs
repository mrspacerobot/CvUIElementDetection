using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Flann;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WindowsFormsApp3
{
    public static class DrawMatches
    {
        public static void FindMatch(Mat modelImage, Mat observedImage, out VectorOfKeyPoint modelKeyPoints, out VectorOfKeyPoint observedKeyPoints, VectorOfVectorOfDMatch matches, out Mat mask, out Mat homography)
        {
            int k = 2;
            double uniquenessThreshold = 0.8 ;

            Stopwatch watch;
            homography = null;

            modelKeyPoints = new VectorOfKeyPoint();
            observedKeyPoints = new VectorOfKeyPoint();

            using (UMat uModelImage = modelImage.GetUMat(AccessType.Read))
            using (UMat uObservedImage = observedImage.GetUMat(AccessType.Read))
            {
                SIFT featureDetector = new SIFT();

                //extract features from the object image
                Mat modelDescriptors = new Mat();
                featureDetector.DetectAndCompute(uModelImage, null, modelKeyPoints, modelDescriptors, false);
                
                watch = Stopwatch.StartNew();

                // extract features from the observed image
                Mat observedDescriptors = new Mat();
                featureDetector.DetectAndCompute(uObservedImage, null, observedKeyPoints, observedDescriptors, false);

                // Bruteforce, slower but more accurate
                // You can use KDTree for faster matching with slight loss in accuracy
                using (Emgu.CV.Flann.LinearIndexParams ip = new Emgu.CV.Flann.LinearIndexParams())
                using (Emgu.CV.Flann.SearchParams sp = new SearchParams())
                using (DescriptorMatcher matcher = new FlannBasedMatcher(ip, sp))
                {
                    matcher.Add(modelDescriptors);

                    matcher.KnnMatch(observedDescriptors, matches, k, null);
                    mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
                    mask.SetTo(new MCvScalar(255));
                    Features2DToolbox.VoteForUniqueness(matches, uniquenessThreshold, mask);

                    int nonZeroCount = CvInvoke.CountNonZero(mask);
                    if (nonZeroCount >= 4)
                    {
                        nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints,
                        matches, mask, 1.5, 20);
                        if (nonZeroCount >= 4)
                        {
                            homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints,
                              observedKeyPoints, matches, mask, 3);
                        }
                    }
                }
                watch.Stop();

            }
            
        }

        public static Mat Draw(Mat modelImage, Mat observedImage)
        {
            Mat homography;
            VectorOfKeyPoint modelKeyPoints;
            VectorOfKeyPoint observedKeyPoints;
            using (VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch())
            {
                Mat mask;
                FindMatch(modelImage, observedImage, out modelKeyPoints, out observedKeyPoints, matches,
                out mask, out homography);

                //Draw the matched keypoints
                Mat result = new Mat();
                Features2DToolbox.DrawMatches(modelImage, modelKeyPoints, observedImage, observedKeyPoints,
                matches, result, new MCvScalar(255, 0,0), new MCvScalar(255, 0,0), mask);

                #region draw the projected region on the image

                if (homography != null)
                {
                    //draw a rectangle along the projected model
                    Rectangle rect = new Rectangle(Point.Empty, modelImage.Size);
                    PointF[] pts = new PointF[]
                    {
                     new PointF(rect.Left, rect.Bottom),
                     new PointF(rect.Right, rect.Bottom),
                     new PointF(rect.Right, rect.Top),
                     new PointF(rect.Left, rect.Top)
                    };
                    pts = CvInvoke.PerspectiveTransform(pts, homography);

                    #if NETFX_CORE
                    Point[] points = Extensions.ConvertAll<PointF, Point>(pts, Point.Round);
                    #else
                    Point[] points = Array.ConvertAll<PointF, Point>(pts, Point.Round);
                    #endif
                    using (VectorOfPoint vp = new VectorOfPoint(points))
                    {
                        CvInvoke.Polylines(result, vp, true, new MCvScalar(0, 255, 0, 255), 5);
                    }
                }
                #endregion

                return result;

            } 

        }
        public static Mat ProcessImage(Mat img)
        {
            
            using (UMat gray = new UMat())
            using (UMat cannyEdges = new UMat())
            using (Mat triangleRectangleImage = new Mat(img.Size, DepthType.Cv8U, 3)) //image to draw triangles and rectangles on
            using (Mat circleImage = new Mat(img.Size, DepthType.Cv8U, 3)) //image to draw circles on
            using (Mat lineImage = new Mat(img.Size, DepthType.Cv8U, 3)) //image to drtaw lines on
            {
                //Convert the image to grayscale and filter out the noise
                CvInvoke.CvtColor(img, gray, ColorConversion.Bgr2Gray);

                //Remove noise
                //CvInvoke.GaussianBlur(gray, gray, new Size(3,3), 1);

                #region circle detection
                double cannyThreshold = 180.0;
                //120 to avoid circles; 28 to find small (radiobuttons)
                double circleAccumulatorThreshold = 28;
                CircleF[] circles = CvInvoke.HoughCircles(gray, HoughModes.Gradient, 1.0, 10.0, cannyThreshold,
                    circleAccumulatorThreshold, 5);
                #endregion

                #region Canny and edge detection
                double cannyThresholdLinking = 120.0;
                CvInvoke.Canny(gray, cannyEdges, cannyThreshold, cannyThresholdLinking);
                LineSegment2D[] lines = CvInvoke.HoughLinesP(
                    cannyEdges,
                    1, //Distance resolution in pixel-related units
                    Math.PI / 45.0, //Angle resolution measured in radians.
                    1, //threshold
                    2, //min Line width
                    10); //gap between lines
                #endregion

                #region Find triangles and rectangles
                List<Triangle2DF> triangleList = new List<Triangle2DF>();
                List<RotatedRect> boxList = new List<RotatedRect>(); //a box is a rotated rectangle
                using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
                {
                    CvInvoke.FindContours(cannyEdges, contours, null, RetrType.List,
                        ChainApproxMethod.ChainApproxSimple);
                    int count = contours.Size;
                    for (int i = 0; i < count; i++)
                    {
                        using (VectorOfPoint contour = contours[i])
                        using (VectorOfPoint approxContour = new VectorOfPoint())
                        {
                            CvInvoke.ApproxPolyDP(contour, approxContour, CvInvoke.ArcLength(contour, true) * 0.04,
                                true);
                            if (CvInvoke.ContourArea(approxContour, false) > 50
                            ) //only consider contours with area greater than 250
                            {
                                if (approxContour.Size == 3) //The contour has 3 vertices, it is a triangle
                                {
                                    Point[] pts = approxContour.ToArray();
                                    triangleList.Add(new Triangle2DF(
                                        pts[0],
                                        pts[1],
                                        pts[2]
                                    ));
                                }
                                else if (approxContour.Size == 4) //The contour has 4 vertices.
                                {
                                    #region determine if all the angles in the contour are within [80, 100] degree
                                    bool isRectangle = true;
                                    Point[] pts = approxContour.ToArray();
                                    LineSegment2D[] edges = PointCollection.PolyLine(pts, true);

                                    for (int j = 0; j < edges.Length; j++)
                                    {
                                        double angle = Math.Abs(
                                            edges[(j + 1) % edges.Length].GetExteriorAngleDegree(edges[j]));
                                        if (angle < 85 || angle > 95)
                                        {
                                            isRectangle = false;
                                            break;
                                        }
                                    }

                                    #endregion

                                    if (isRectangle)
                                    {
                                        boxList.Add(CvInvoke.MinAreaRect(approxContour));
                                    }
                                }
                            }
                        }
                    }
                }
                #endregion

                #region draw triangles and rectangles
                triangleRectangleImage.SetTo(new MCvScalar(0));
                foreach (Triangle2DF triangle in triangleList)
                {
                    CvInvoke.Polylines(triangleRectangleImage, Array.ConvertAll(triangle.GetVertices(), Point.Round),
                        true, new Bgr(Color.DarkBlue).MCvScalar, 2);
                }

                foreach (RotatedRect box in boxList)
                {
                    CvInvoke.Polylines(triangleRectangleImage, Array.ConvertAll(box.GetVertices(), Point.Round), true,
                        new Bgr(Color.DarkOrange).MCvScalar, 2);
                }

                //Drawing a light gray frame around the image
                CvInvoke.Rectangle(triangleRectangleImage,
                    new Rectangle(Point.Empty,
                        new Size(triangleRectangleImage.Width - 1, triangleRectangleImage.Height - 1)),
                    new MCvScalar(120, 120, 120));
                //Draw the labels
                CvInvoke.PutText(triangleRectangleImage, "Triangles and Rectangles", new Point(20, 20),
                    FontFace.HersheyDuplex, 0.5, new MCvScalar(120, 120, 120));
                #endregion

                #region draw circles
                circleImage.SetTo(new MCvScalar(0));
                foreach (CircleF circle in circles)
                    CvInvoke.Circle(circleImage, Point.Round(circle.Center), (int)circle.Radius,
                        new Bgr(Color.Brown).MCvScalar, 2);

                //Drawing a light gray frame around the image
                CvInvoke.Rectangle(circleImage,
                    new Rectangle(Point.Empty, new Size(circleImage.Width - 1, circleImage.Height - 1)),
                    new MCvScalar(120, 120, 120));
                //Draw the labels
                CvInvoke.PutText(circleImage, "Circles", new Point(20, 20), FontFace.HersheyDuplex, 0.5,
                    new MCvScalar(120, 120, 120));
                #endregion

                #region draw lines
                lineImage.SetTo(new MCvScalar(0));
                foreach (LineSegment2D line in lines)
                    CvInvoke.Line(lineImage, line.P1, line.P2, new Bgr(Color.Green).MCvScalar, 2);
                //Drawing a light gray frame around the image
                CvInvoke.Rectangle(lineImage,
                    new Rectangle(Point.Empty, new Size(lineImage.Width - 1, lineImage.Height - 1)),
                    new MCvScalar(120, 120, 120));
                //Draw the labels
                CvInvoke.PutText(lineImage, "Lines", new Point(20, 20), FontFace.HersheyDuplex, 0.5,
                    new MCvScalar(120, 120, 120));
                #endregion

                Mat result = new Mat();
                CvInvoke.VConcat(new Mat[] { img, triangleRectangleImage, circleImage, lineImage }, result);
                return result;
            }
        }
    }
}
