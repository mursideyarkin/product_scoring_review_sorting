###################################################
# PRODUCT SCORING & REVIEW SORTING: AMAZON CASE
###################################################

# PROJECT STEPS:
# GOAL 1: CALCULATING PRODUCT SCORE
    # STEP 1: Import dataset & libraries
    # STEP 2: Select the product with the most comments
    # STEP 3: Calculate average score of the product
    # STEP 4: Calculate the weighted average score by date

# GOAL 2: DETERMINE THE FIRST 20 COMMENTS TO BE DISPLAYED ON THE PRODUCT PROMOTION PAGE
    # STEP 1: Extracting number of helpful & not helpful votes for each comment
    # STEP 2: Calculating simple comment score
    # STEP 3: Calculating average rating score
    # STEP 4: Scoring with Wilson lower bound method
    # STEP 5: Selecting the first 20 comments to be displayed on the product promotion page

#####################################
# GOAL 1: CALCULATING PRODUCT SCORE
#####################################

###################################################
# STEP 1: Import dataset & libraries
###################################################
import pandas as pd
from scipy import stats as st
import math

pd.set_option('display.max_columns', None)

df = pd.read_csv("6th_week/homeworks/df_sub.csv")

###################################################
# STEP 2: Select the product with the most comments
###################################################

max_comment = df.groupby("asin").count().sort_values("overall", ascending=False).head(1).index
df_sub = df[df["asin"] == max_comment[0]]
df_sub.head()

###################################################
# STEP 3: Calculate average score of the product
###################################################

df_sub["overall"].mean() #4.587

#######################################################
# STEP 4: Calculate the weighted average score by date
#######################################################

# How many days have passed since the comment (day_diff):
df_sub["reviewTime"] = pd.to_datetime(df_sub["reviewTime"], dayfirst=True)
current_date = pd.to_datetime("2014-12-08 0:0:0")
df_sub["day_diff"] = (current_date - df_sub["reviewTime"]).dt.days

# Divide day_diff into three time groups:
t1 = df_sub["day_diff"].quantile(0.25)
t2 = df_sub["day_diff"].quantile(0.50)
t3 = df_sub["day_diff"].quantile(0.75)

# Calculate the weighted score according to t1, t2, t3 values:

weighted_avg_time = df.loc[df_sub["day_diff"] <= t1, "overall"].mean() * 0.28 + \
                    df.loc[(df_sub["day_diff"] > t1) & (df_sub["day_diff"] <= t2), "overall"].mean() * 0.26 + \
                    df.loc[(df_sub["day_diff"] > t2) & (df_sub["day_diff"] <= t3), "overall"].mean() * 0.24 + \
                    df.loc[df_sub["day_diff"] > t3, "overall"].mean() * 0.22

# weighted average product score is 4.595.
# average product score is 4.587.

#######################################################################################
# GOAL 2: DETERMINE THE FIRST 20 COMMENTS TO BE DISPLAYED ON THE PRODUCT PROMOTION PAGE
#######################################################################################

# STEP 1: Extracting number of helpful & not helpful votes for each comment
# STEP 2: Calculating simple comment score
# STEP 3: Calculating average rating score
# STEP 4: Scoring with Wilson lower bound method
# STEP 5: Selecting the first 20 comments to be displayed on the product promotion page

#############################################################################
# STEP 1: Extracting number of helpful & not helpful votes for each comment
#############################################################################

df_sub["helpful"].head()
# "helpful" variable includes two values:
# first value represents the number of votes that found the comment helpful.
# second value represents the number of total votes including helpful and not helpful.
# We'll calculate the helpful_no by separating the two values and then applying (total_vote - helpful_yes):

# new data frame with split value columns
new = df_sub["helpful"].str.split(",", n=1, expand=True)

# making separate first name column from new data frame
df_sub["helpful_yes"] = new[0]
df_sub["helpful_yes"] = df_sub["helpful_yes"].map(lambda x: x.lstrip('[')).astype("float")
df_sub["total_vote"] = new[1]
df_sub["total_vote"] = df_sub["total_vote"].map(lambda x: x.rstrip(']')).astype("float")
df_sub["helpful_no"] = df_sub["total_vote"] - df_sub["helpful_yes"]

# Cross-check for the new variables:
df_sub["total_vote"].sum() == df_sub["helpful_yes"].sum() + df_sub["helpful_no"].sum()

###################################################
# STEP 2: Calculating simple comment score
###################################################

# score_pos_neg_diff = postivie votes - negatif votes

df_sub["score_pos_neg_diff"] = df_sub["helpful_yes"] - df_sub["helpful_no"]

###################################################
# STEP 3: Calculating average rating score
###################################################

def score_average_rating(pos, neg):
    if pos + neg == 0:
        return 0
    return pos / (pos + neg)

# score_average_rating
df_sub["score_average_rating"] = df_sub.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

df_sub.sort_values("score_average_rating", ascending=False).head()

##################################################
# STEP 4: Scoring with Wilson lower bound method
###################################################

def wilson_lower_bound(pos, neg, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not: Eğer skorlar 1-5 arasıdaysa 1-3 down, 4-5 up olarak işaretlenir ve bernoulli'ye uygun hale getirilir.

    Parameters
    ----------
    pos: int
        pozitif yorum sayısı
    neg: int
        negatif yorum sayısı
    confidence: float
        güven aralığı

    Returns
    -------
    wilson score: float

    """
    n = pos + neg
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df_sub["wilson_lower_bound"] = df_sub.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df_sub.sort_values("wilson_lower_bound", ascending=False).head()

#######################################################################################
# STEP 5: Selecting the first 20 comments to be displayed on the product promotion page
#######################################################################################

# Since the wilson lower bound method provides us a confidence interval,
# we selected the top 20 comments based on the wilson lower bound score.
df_sub[["helpful_yes","helpful_no","total_vote","score_pos_neg_diff", "score_average_rating","wilson_lower_bound"]].sort_values("wilson_lower_bound", ascending=False).head(20)

# When we look at the comment with the highest wilson_lower_bound score,
# we observe that the score_average_rating and score_pos_neg_diff values are also quite high.

# Let's compare the 3449 indexed comment with the 4212 indexed comment:
# The 3449 indexed comment has a higher wilson_lower_bound score, it ranked second. The other comment ranked as 3rd.
# Less votes were given to 3449 indexed reviews, and more votes were given for 4212 indexed reviews.
# Although the 3449 indexed comments received less votes, it was found much more useful than the 4212 indexed comment.
# Therefore, wilson_lower_bound score is calculated higher.