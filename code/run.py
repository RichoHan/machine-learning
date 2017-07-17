#!/usr/bin/python
import pandas as pd
from task_io import TaskIO

SPAM_FEATURES = [
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

SELECTED_FEATURES = [
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

    def export_data(self, data, path):
        data.to_csv(path, index=False)

    def export_prediction(self, X_test, prediction):
        submission = pd.concat(
            [
                pd.Series([i + 1 for i in range(len(X_test))]),
                pd.Series(prediction).astype(float)
            ],
            axis=1
        )
        submission.columns = ['id', 'value']
        super(Session2TaskIO, self).export_prediction(submission)

    def export_recording(self, costs, scores, path):
        exp = pd.concat(
            [
                pd.Series(range(1, len(costs) + 1)),
                pd.Series(costs).astype(float),
                pd.Series(scores).astype(float)
            ],
            axis=1
        )
        exp.columns = ['round', 'cost', 'score']
        self.export_data(exp, path)


def cross_validation(X, y, score_func, splits=10, _lambda=0):
    """ Run K-fold validation and return prediction scores.

    Parameters
    ----------
    X (ndarray): Training vector.

    y (ndarray): Target vector.

    Returns
    -------
    scores : Series, shape = (n_samples,)
        score of from each validation round.
    """
    from sklearn.model_selection import StratifiedKFold
    from linear_model import LogisticRegression
    scores = list()

    kf = StratifiedKFold(n_splits=splits, shuffle=True)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = LogisticRegression()
        model.fit(X_train, y_train, _lambda)
        scores.append(score_func(y_test, model.predict(X_test)))

    # return pd.Series(scores, index=range(1, splits + 1))
    return scores


def main():
    # ===== Import training and testing data =====
    # Read training and testing data, and then separate features from answers for cross-validation.
    task_io = Session2TaskIO(
        train='./data/spam_train.csv',
        test='./data/spam_test.csv',
        result='./data/submission.csv',
        validation='./data/validation.csv'
    )

    column_names = ['id'] + SPAM_FEATURES + ['spam']
    training_data = task_io.import_training_data(names=column_names)
    X = training_data.loc[:, SELECTED_FEATURES]
    y = training_data.loc[:, 'spam']

    # # ===== Predict testing data =====
    # column_names = ['id'] + SPAM_FEATURES
    # testing_data = task_io.import_testing_data(names=column_names)

    # from linear_model import LogisticRegression
    # model = LogisticRegression()
    # model.fit(X, y, 1000)

    # X_test = testing_data.loc[:, SELECTED_FEATURES]
    # y_test = pd.Series([0 for i in range(len(X_test))])
    # prediction = model.predict(X_test)
    # task_io.export_prediction(X_test, prediction)

    # costs, scores = model.get_recording(X_test, y_test, task_io.score)
    # task_io.export_recording(costs, scores, './data/submission_exp.csv')

    # # ===== Cross validation =====
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)

    # from linear_model import LogisticRegression
    # model = LogisticRegression()
    # model.fit(X_train, y_train)

    # costs, scores = model.get_recording(X_test, y_test, task_io.score)
    # task_io.export_recording(costs, scores, './data/validation_exp.csv')
    # print('Score: {0}'.format(task_io.score(y_test, model.predict(X_test))))

    # ===== K-fold validation =====
    n_splits = 10
    data = list()
    for i in range(0, 6):
        scores = cross_validation(X, y, task_io.score, n_splits, 10**i)
        for idx, score in enumerate(scores):
            validation = data.append([idx + 1, 10**i, score])
    validation = pd.DataFrame(data, columns=['trial', 'regularization', 'score'])
    task_io.export_validation(validation)


if __name__ == "__main__":
    main()
