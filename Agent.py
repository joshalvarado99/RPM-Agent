import PIL
import numpy as np
from RavensProblem import RavensProblem
import cv2

class Agent:
    def __init__(self):
        pass

    def Solve(self, problem):
        figures = self.load_and_preprocess_images(problem)

        if problem.problemType == '2x2':
            training_pairs = [('horizontal', 'A', 'B'),
            ('vertical', 'A', 'C')]
            answer_choices = range(1, 7)
            training_transformations = self.assess_training_transformations(figures, training_pairs)

            similarity_scores = self.calculate_similarity_scores(figures, training_transformations, answer_choices)

            best_answer = self.find_best_answer(similarity_scores)
            return best_answer
        else:
            return -1

    def assess_training_transformations(self, figures, training_pairs):
        training_transformations = {}
        for relation, fig1, fig2 in training_pairs:
            image1 = figures[fig1]
            image2 = figures[fig2]
            transformations = self.Calculate_Transformations(image1, image2)
            training_transformations[relation] = transformations
        return training_transformations

    def calculate_similarity_scores(self, figures, training_transformations, answer_choices):
        similarity_scores = [0] * 6

        for answer in answer_choices:
            answer_image = figures[str(answer)]

            # Horizontal comparison
            transformations_horizontal = self.Calculate_Transformations(figures['C'], answer_image)
            similarity_horizontal = self.Compare_Transformations(
                training_transformations['horizontal'], transformations_horizontal)

            # Vertical comparison
            transformations_vertical = self.Calculate_Transformations(figures['B'], answer_image)
            similarity_vertical = self.Compare_Transformations(
                training_transformations['vertical'], transformations_vertical)

            # Sum both
            total_similarity = similarity_horizontal + similarity_vertical
            similarity_scores[answer - 1] = total_similarity

        return similarity_scores

    def find_best_answer(self, similarity_scores):
        return similarity_scores.index(max(similarity_scores)) + 1
    

    #Referencing open CV documentation: https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
    #Modified code for this particular problem but documentation was used a reference
    #Begining open cv code
    def load_and_preprocess_images(self, problem):
        figures = {}
        for name, figure in problem.figures.items():
            image = cv2.imread(figure.visualFilename, 0)
            image = self.Binarize_Image(image)
            figures[name] = image
        return figures

    def Binarize_Image(self, image):
        extra, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY) #Using threshold values from opencv documentation: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html 
        return image
    #Ending of open cv code

    def Rotate_Image(self, image, angle):
        return self.Rotate_Image(image, angle)

    def Flip_Image(self, image, flip):
        if flip == 'horizontal':
            return cv2.flip(image, 1)
        elif flip == 'vertical':
            return cv2.flip(image, 0)
        else:
            return image

    def Calculate_Similarity_For_Transformation(self, transformed_image, image2, angle, flip):
        score = self.Calculate_Image_Similarity(transformed_image, image2)
        return {
            'rotation': angle,
            'flip': flip,
            'similarity': score
        }

    def Calculate_Transformations(self, image1, image2):
        transformations = []
        rotation_angles = [0, 90, 180, 270]
        flips = [None, 'horizontal', 'vertical']

        # Iterate through all possible rotations and flips
        for angle in rotation_angles:
            rotated_image = self.Rotate_Image(image1, angle)
            for flip in flips:
                transformed_image = self.Flip_Image(rotated_image, flip)
                transformation = self.Calculate_Similarity_For_Transformation(transformed_image, image2, angle, flip)
                transformations.append(transformation)

        return transformations

    def Rotate_Image(self, image, angle):
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    #I used this code from CV Notes as a reference with some slight modifications: https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
    #Beginning of code from CV Notes
    def Calculate_Image_Similarity(self, image1, image2):
        # Ensure images are the same size
        if image1.shape != image2.shape:
            return 0
        ssim_score = self.SSIM(image1, image2)
        return ssim_score

    def SSIM(self, img1, img2):
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        mu1 = img1.mean()
        mu2 = img2.mean()
        sigma1_sq = ((img1 - mu1) ** 2).mean()
        sigma2_sq = ((img2 - mu2) ** 2).mean()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim = numerator / denominator
        return ssim
    #End of code from CV Notes

    def Compare_Transformations(self, transformations1, transformations2):
        max_similarity = 0
        for trans1 in transformations1:
            for trans2 in transformations2:
                if trans1['rotation'] == trans2['rotation'] and trans1['flip'] == trans2['flip']:
                    # Average SSIM
                    similarity = (trans1['similarity'] + trans2['similarity']) / 2
                    if similarity > max_similarity:
                        max_similarity = similarity
        return max_similarity