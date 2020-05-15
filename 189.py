import numpy as np
import base64

"""
By Ellek Linton

This is just preliminary and accuracy cannot be guarenteed (but it should
provide an accurate lower-bound on your grade).

This is based upon the grade bins from Spring 2019 as provided by Professor Shewchuck in @974 on Piazza

This computes your grade using either your midterm A score, or your re-normalized midterm B score.
If you took midterm A, leave midterm B as 0/100.
If you took midterm B, leave midterm A as 0/100

It also weights all homeworks equally as specified by the course policy.

For the undergraduate grading scheme, set use_grad_grading_scheme to False

To use the graduate grading scheme, set use_grad_grading_scheme to True
and grad_project_score to your estimated score on the graduate project.

If you are bowing out of the midterm, set midterm_bow_out to True. Otherwise, set it to False.

******* INSTRUCTIONS *******
To use, replace CHANGE_THIS with the proper value!

"""

homework_scores = {
    "hw1" : 26/26, #https://www.gradescope.com/courses/84413/assignments/338700
    "hw2" : 63.5/64, #https://www.gradescope.com/courses/84413/assignments/342164
    "hw3" : 52/57, #https://www.gradescope.com/courses/84413/assignments/360361
    "hw4" : 57.5/60, #https://www.gradescope.com/courses/84413/assignments/398749
    "hw5" : 60/61, #https://www.gradescope.com/courses/82897/assignments/389569
    "hw6" : 129/129, #https://www.gradescope.com/courses/84413/assignments/436549
    "hw7" : 100/100, #https://www.gradescope.com/courses/82897/assignments/457863
}

### Put zero for the midterm that you did not take
midterm_a_score = 0/100

midterm_b_score = 57.5/100

# If you did not take the midterm and are replacing it with the final, set the following to True
# If you DID take the midterm, set the following to False.
midterm_bow_out = False # True or False

use_grad_grading_scheme = False # True or False
grad_project_score = 0/100 #Estimated grad project score



#################################### Don't edit anything below! ####################################

grades = ["A+", "A", "A-", "B-", "C-", "F"]
gradesPNP = ["P", "P", "P", "P", "P", "NP"]
gradeBins = [82, 70, 66, 53, 38, 0]

print("-" * 50)


def midterm_score():
    if midterm_bow_out:
        return 0

    if midterm_a_score != 0:
        return midterm_a_score

    midterm_b_z_score = ((midterm_b_score * 100) - 42.396) / 19.705

    translated_score = 55.1 + (16.11 * midterm_b_z_score)
    return max(0, translated_score / 100)


def cumulative_score():
    hw_component = np.average(list(homework_scores.values())) * 0.4
    if midterm_bow_out:
        midterm_component = 0
    else:
        midterm_component = midterm_score() * 0.2

    grad_project_component = 0

    if use_grad_grading_scheme:
        grad_project_component = grad_project_score * 0.2

    return (grad_project_component + midterm_component + hw_component) * 100


def gradeBin(score=cumulative_score()):
    bin = 0
    while score < gradeBins[bin]:
        bin += 1
    return bin


def currentStatus():
    bin = gradeBin()
    if gradesPNP[bin] == "P":
        print("Congrats! You already have enough points to pass CS189 EVEN IF you got a zero on the final!")
        print("Now relax and unwind because you will pass even if you literally bomb the final.")
        print("-" * 50)
        print(
            f"EVEN IF you score ZERO on the final, your cumulative grade would be at least {grades[bin]} ({gradesPNP[bin]}) ({round(cumulative_score(), 2)}%) in CS189.")
        if not midterm_bow_out:
            final_weight = 0.4
            if use_grad_grading_scheme:
                final_weight = 0.2
            print("-" * 50)
            midterm_predicted_final_score = cumulative_score() + midterm_score() * 100 * final_weight
            midter_predicted_bin = gradeBin(midterm_predicted_final_score)
            print(
                f"If you score on the final exactly how you scored on the midterm, your final grade would be at least {grades[midter_predicted_bin]} ({gradesPNP[midter_predicted_bin]}) ({round(midterm_predicted_final_score, 2)}%) in CS189.")
    else:
        print(
            f"You currently have a {grades[bin]} ({gradesPNP[bin]}) in CS189. You still have the potential to pass though!")
    print("-" * 50)
    m = 'SW4gdGhlIHdvcmRzIG9mIEthbnllLCAiS2VlcCB5b3VyIG5vc2Ugb3V0IHRvIHRoZSBza3ksIGtlZXAgeW91ciBoZWFydCB0byBnb2QsIGFuZCBrZWVwIHlvdXIgZmFjZSB0byB0aGUgcmlzaW5nIHN1bi4i'
    print(str(base64.b64decode(m))[2:-1])


def possibleFinalGrades():
    curr_score = cumulative_score()
    for i in list(range(len(grades) - 1))[::-1]:
        binCutoff = gradeBins[i]
        binGrade = grades[i]
        binPassing = gradesPNP[i]
        if use_grad_grading_scheme:
            final_weight = 0.2
        else:
            final_weight = 0.4

        if midterm_bow_out:
            final_weight += 0.2

        final_percent_needed = max(0, (binCutoff - curr_score) / final_weight)
        if final_percent_needed > 0:
            print(
                f"To get at least {binGrade} ({binPassing}) in CS189, you need to get {round(final_percent_needed, 2)}% on the final exam!")
        else:
            print(f"You already have enough points to get at least {binGrade} ({binPassing}) in CS189! Yay lol.")


currentStatus()
print("-" * 50)
print("Now, here is what you need to do on the final to get the following final course grades:\n")
possibleFinalGrades()
m = 'WW91IGdvdCBkaXMsIGRvbid0IGxvc2UgaG9wZS4gSnVzdCByZW1lbWJlciB5b3UgZ28gdG8gQkVSS0VMRVkgd2hpY2ggaXMgdGhlIGdyZWF0ZXN0IHNjaG9vbCBpbiB0aGUgd29ybGQgYW5kIHlvdSBhcmUgS0lMTElORyBJVCE='
print("\n")
print(str(base64.b64decode(m))[2:-1])
