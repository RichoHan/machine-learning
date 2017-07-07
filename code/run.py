#!/usr/bin/python
import pandas as pd
from task_io import TaskIO

FEATURES = [
    # Word frequency.
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
    'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
    'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
    'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
    'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
    'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money',
    'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650',
    'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857',
    'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology',
    'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
    'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
    'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference',

    # Frequency of special characters.
    'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#',

    # Continuous sequence of CAPITALS.
    'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total',
]


class Session2TaskIO(TaskIO):
    def get_processed_training(self):
        pass

    def get_processed_testing(self):
        pass

    def score(self, y_test, y_predicted):
        return (y_test == y_predicted).value_counts(True)[1]


def cross_validation(X, y, score_func, splits=10):
    from sklearn.model_selection import KFold
    from linear_model import LogisticRegression
    scores = list()

    kf = KFold(n_splits=splits, shuffle=True)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = LogisticRegression()
        model.fit(X_train, y_train)
        scores.append(score_func(y_test, model.predict(X_test)))

    return pd.Series(scores, index=range(1, splits + 1))


def main():
    # ===== Import training and testing data =====
    task_io = Session2TaskIO(
        train='./data/spam_train.csv',
        test='./data/spam_test.csv',
        result='./data/submission.csv',
        validation='./data/validation.csv'
    )

    training_data = task_io.import_training_data(names=['id'] + FEATURES + ['spam'])
    X = training_data.loc[:, 'id':'capital_run_length_total']
    y = training_data.loc[:, 'spam']

    # ===== Cross validation =====
    validation = pd.DataFrame(columns=range(1, 11))
    scores = cross_validation(X, y, task_io.score)
    validation = validation.append([scores], ignore_index=True)
    task_io.export_validation(validation)

    # ===== Predict testing data =====
    # testing_data = task_io.import_testing_data(names=['id'] + FEATURES)

    # from linear_model import LogisticRegression
    # model = LogisticRegression()
    # model.fit(X, y)
    # prediction = model.predict(testing_data)
    # submission = pd.concat(
    #     [
    #         testing_data.loc[:, 'id'].reset_index(drop=True).map(lambda x: x + 1),
    #         pd.Series(prediction).astype(float)
    #     ],
    #     axis=1
    # )
    # submission.columns = ['id', 'value']
    # task_io.export_prediction(submission)


if __name__ == "__main__":
    main()
