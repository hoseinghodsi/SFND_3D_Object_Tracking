
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <random>
#include <iterator>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, std::string imgPath, cv::Size worldSize, cv::Size imageSize, bool saveImg, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }


    cv::Mat resizedTopViewImg;

    cv::resize(topviewImg, resizedTopViewImg, cv::Size(), 0.75, 0.75);
    if (saveImg) cv::imwrite(imgPath, resizedTopViewImg);
    
    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBoxCurr, BoundingBox &boundingBoxPrev, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                    std::vector<cv::DMatch> &kptMatches, double shrinkPrct)
{
    double meanDistVal = 0;

    std::vector<cv::DMatch> kptMatchesROI;
    double minRoiDim;
    double roiRadius;
    

    minRoiDim = std::min(boundingBoxCurr.roi.width, boundingBoxCurr.roi.height);
    roiRadius = std::sqrt(std::pow((minRoiDim / 2.0), 2));

    for (auto it = kptMatches.begin(); it != kptMatches.end(); ++it)
    {
        double kptX = (boundingBoxCurr.roi.x + boundingBoxCurr.roi.width / 2.0) - kptsCurr[it->trainIdx].pt.x;
        double kptY = (boundingBoxCurr.roi.y + boundingBoxCurr.roi.height/ 2.0) - kptsCurr[it->trainIdx].pt.y;

        double kptRadius = std::sqrt(std::pow(kptX, 2) + std::pow(kptY, 2));
       
        if (kptRadius < roiRadius*(1-shrinkPrct))
        {
            kptMatchesROI.push_back(*it);
        }

    }
    for (auto it = kptMatchesROI.begin(); it != kptMatchesROI.end(); ++it)
    {
        meanDistVal += it->distance;
    }

    if (kptMatchesROI.size() > 0)
    {
        meanDistVal = meanDistVal/kptMatchesROI.size();
    }
    else return;

    
    for (auto it = kptMatchesROI.begin(); it != kptMatchesROI.end(); ++it)
    {
        
        cv::KeyPoint kp1;
        cv::KeyPoint kp2;
        boundingBoxCurr.kptMatches.push_back(*it);
            
        kp1.pt = cv::Point(kptsCurr[it->trainIdx].pt);
        kp2.pt = cv::Point(kptsCurr[it->queryIdx].pt);
        kp1.size = 10.0;
        kp2.size = 10.0;
        boundingBoxCurr.keypoints.push_back(kp1);
        boundingBoxPrev.keypoints.push_back(kp2);

    }

}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // computesdistance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    double sumDistRatio = 0;

    if (kptMatches.size() > 0)
    {
        for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
        {
            cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
            cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

            for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
            { 
                double minDist = 100.0; // min. required distance

                // get next keypoint and its matched partner in the prev. frame
                cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
                cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);
                // compute distances and distance ratios
                double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
                double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

                if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
                {
                    double distRatio = distCurr / distPrev;
                    distRatios.push_back(distRatio);
                }
            } 
        }
    }
    else
    {
        //TTC = 0;
        TTC = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    // Sorting the keypoints and filtering the outliers using the median distance ratio
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    TTC = (-1.0 / frameRate) / (1 - medDistRatio);
}

void lidarPointsProcessing(std::vector<LidarPoint> &lidarPoints, bool doSample=false, int samplingPoints=200)
{
    //cout << "pre size: " << lidarPoints.size() << endl;
    
    // Performing point sampling if it is turned on
    if (doSample && lidarPoints.size() > samplingPoints)
    {
        std::random_shuffle(lidarPoints.begin(), lidarPoints.end());
        lidarPoints.erase(lidarPoints.begin()+samplingPoints, lidarPoints.end());
        //cout << "pre size: " << lidarPoints.size() << endl;
    }

    // sorting the lidar points based on their distance
    std::sort(lidarPoints.begin(), lidarPoints.end(), [](LidarPoint smP, LidarPoint lrP)
    {
        return smP.x < lrP.x;
    });
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // taking a sample out of the lidar points vector
    bool doSampling = true;
    int samplingPoints = 150; // how many samples

    // calls the helper function to proccess the sampling and sorting of the lidar point vectors
    lidarPointsProcessing(lidarPointsPrev, doSampling, samplingPoints);
    lidarPointsProcessing(lidarPointsCurr, doSampling, samplingPoints);
    
    // finds the distance in prev and curr frame based on the x-coord median of the lidar points 
    double d0 = lidarPointsPrev[lidarPointsPrev.size()/2].x;
    double d1 = lidarPointsCurr[lidarPointsCurr.size()/2].x;

    // calculates the TTC in seconds
    TTC = d1 * (1.0 / frameRate) / (d0 - d1);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // ...
    // Initializing a multimap to store all the boundind boxes IDs
    // The multimap key and element components are the current- and previous-frame bounding boxes IDs, respectively.
    // Using multimap to be able to store multiple elements with the same keys.

    std::multimap<int, int> BIGmmap {};
    int maxPrevBoundingBoxIds = 0;
    for (auto match : matches)
    {
        // extracting the keypoint in the previous and current frames 
        cv::KeyPoint prevKeyPoint = prevFrame.keypoints[match.queryIdx];
        cv::KeyPoint currKeyPoint = currFrame.keypoints[match.trainIdx];

        std::vector<int> previousBoxId;
        std::vector<int> currentBoxId;

        // Check all the bounding boxes in the previous frame
        // generates a vector of box ids containing the match point
        bool queryPtFound = false;
        for (auto bb : prevFrame.boundingBoxes)
        {
            if (bb.roi.contains(prevKeyPoint.pt))
            {
                queryPtFound = true;
                previousBoxId.push_back(bb.boxID);
            }
        }

        // Check all the bounding boxes in the current frame
        // generates a vector of box ids containing the match point
        bool trainPtFound = false;
        for (auto bb : currFrame.boundingBoxes)
        {
            if (bb.roi.contains(currKeyPoint.pt))
            {
                trainPtFound = true;
                currentBoxId.push_back(bb.boxID);
            }
        }
        
        if (queryPtFound && trainPtFound)
        {
            int minFound = std::min(previousBoxId.size(), currentBoxId.size());
            for (int i=0; i<minFound; i++)
            {
                BIGmmap.insert({currentBoxId[i], previousBoxId[i]});
                maxPrevBoundingBoxIds = std::max(maxPrevBoundingBoxIds, previousBoxId[i]);    
            }
                  
        }
    }
    // Generating a list of box Ids of current frames
    vector<int> currFrameBoxIds {};

    for (auto box : currFrame.boundingBoxes)
    {
        currFrameBoxIds.push_back(box.boxID);
    }  

    for (int ii : currFrameBoxIds)
    {
        int max = 0;
        int maxIndex = -1;

        // finding a range of Ids from previous frame mapping to current frame Id
        auto crspPrevIds = BIGmmap.equal_range(ii);

        std::vector<int> prevCountTracker(maxPrevBoundingBoxIds+1, 0);
        std::vector<int> prevIndexTracker(maxPrevBoundingBoxIds+1, -1);

        for (auto it = crspPrevIds.first; it != crspPrevIds.second; ++it)
        {        
            prevCountTracker[(*it).second] += 1;
            prevIndexTracker[(*it).second] = (*it).second;

        }

        int cnt = 0;
        for (int i=0; i < prevCountTracker.size(); ++i)
        {
            cnt = prevCountTracker[i];
            if (cnt>max)
            {
                max = cnt;
                maxIndex = prevIndexTracker[i];
            }
            
        }
        
        // Inserting the best matching box map
        // key: previous frame bounding box ID most likely to be the match 
        // val: current frame bounding box ID
        if (maxIndex >= 0)
            bbBestMatches.insert({maxIndex, ii});

    }

} 
