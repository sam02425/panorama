# import the necessary packages
import numpy as np
import imutils
import cv2

class Stitcher:
    def __init__(self):
        # determine if we are using OpenCV v3.X and initialize the
        # cached homography matrix
        self.isv3 = imutils.is_cv3(or_better=True)
        self.cachedH = None



    def stitch(self, images, ratio=0.75, reprojThresh=4.0):
        # unpack the images
        imageA, imageB, imageC = images

        # stitch the left and middle images
        stitchedAB = self.stitch_pair(imageA, imageB, ratio, reprojThresh)

        if stitchedAB is None:
            print("Failed to stitch left and middle images. Stitching with right image only.")
            stitchedAB = imageA  # Use only the left image if stitching fails

        # stitch the stitched left-middle image with the right image
        stitchedABC = self.stitch_pair(stitchedAB, imageC, ratio, reprojThresh)

        if stitchedABC is None:
            print("Failed to stitch the stitched image with the right image. Using the stitched left-middle image only.")
            stitchedABC = stitchedAB  # Use only the stitched left-middle image if stitching with the right image fails

        # return the stitched image
        return stitchedABC

    def stitch_pair(self, imageA, imageB, ratio, reprojThresh):
        # detect keypoints and extract features
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # if the match is None, then there aren't enough matched keypoints to create a panorama
        if M is None:
            return None

        # unpack the matches, homography matrix, and status of matched keypoints
        (matches, H, status) = M

        # check if the homography matrix is valid
        if H is None or H.shape != (3, 3):
            print("Invalid homography matrix. Skipping stitching.")
            return None

        # check if the homography matrix is of the correct data type
        if H.dtype != np.float32:
            H = H.astype(np.float32)

        # apply a perspective warp to stitch the images together
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # return the stitched image
        return result

    # def stitch(self, images, ratio=0.75, reprojThresh=4.0):
    #     # unpack the images
    #     imageA, imageB, imageC = images

    #     # stitch the left and middle images
    #     stitchedAB = self.stitch_pair(imageA, imageB, ratio, reprojThresh)

    #     if stitchedAB is None:
    #         print("Failed to stitch left and middle images. Stitching aborted.")
    #         return None

    #     # stitch the stitched left-middle image with the right image
    #     stitchedABC = self.stitch_pair(stitchedAB, imageC, ratio, reprojThresh)

    #     if stitchedABC is None:
    #         print("Failed to stitch the stitched left-middle image with the right image. Stitching aborted.")
    #         return None

    #     # return the stitched image
    #     return stitchedABC

    # def stitch_pair(self, imageA, imageB, ratio, reprojThresh):
    #     # detect keypoints and extract features
    #     (kpsA, featuresA) = self.detectAndDescribe(imageA)
    #     (kpsB, featuresB) = self.detectAndDescribe(imageB)

    #     # match features between the two images
    #     M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

    #     # if the match is None, then there aren't enough matched keypoints to create a panorama
    #     if M is None:
    #         return None

    #     # otherwise, apply a perspective warp to stitch the images together
    #     (matches, H, status) = M
    #     result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    #     result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

    #     # return the stitched image
    #     return result

    # def stitch_pair(self, imageA, imageB, ratio, reprojThresh):
    #     print("imageA shape:", imageA.shape)
    #     print("imageB shape:", imageB.shape)
    #     # if the cached homography matrix is None, then we need to
    #     # apply keypoint matching to construct it
    #     if self.cachedH is None:
    #         # detect keypoints and extract features
    #         (kpsA, featuresA) = self.detectAndDescribe(imageA)
    #         (kpsB, featuresB) = self.detectAndDescribe(imageB)

    #         # match features between the two images
    #         M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

    #         # if the match is None, then there aren't enough matched
    #         # keypoints to create a panorama
    #         if M is None:
    #             return None

    #         # cache the homography matrix
    #         self.cachedH = M[1]

    #     # apply a perspective transform to stitch the images together
    #     # using the cached homography matrix
    #     result = cv2.warpPerspective(imageA, self.cachedH, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    #     result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # return the stitched image
        return result

    def detectAndDescribe(self, image):
    # convert the image to grayscale
          gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# check if the image is empty
          if gray.size == 0:
               raise ValueError("Empty input image")

		# detect and extract features from the image
          descriptor = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
          (kps, features) = descriptor.detectAndCompute(image, None)

		# # convert the keypoints from KeyPoint objects to NumPy arrays
        #   kps = np.float32([kp.pt for kp in kps])
          if kps is None or features is None:
              print("No keypoints detected")
              return (None, None)

		# return a tuple of keypoints and features
          print("Number of keypoints detected: ", len(kps))
          print("Features shape: ", features.shape)
          return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each other
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i].pt for (_, i) in matches])
            ptsB = np.float32([kpsB[i].pt for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # return the matches along with the homograpy matrix and status of each matched point
            return (matches, H, status)

        # otherwise, no homography could be computed
        return None