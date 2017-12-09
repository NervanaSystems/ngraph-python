import numpy as np
import pandas as pd
# To normalize the data.
from sklearn import preprocessing
import sys


def categorical_numbers(column):
    column_category = column.astype('category')
    return (column_category.cat.codes)


# A function to convert text into hash codes using buscket size.


def sparse_column_with_hash_bucket(text, bucket_size):
    return (hash(text) % bucket_size)


def prodm(l1, l2):
    l = 0

    for x, y in zip(l1, l2):
        l += (x * y)

    return l


def cross_columns(columns):
    cross = columns
    cross_frame = pd.concat(cross, axis=1)
    cardinality = []
    cardinality_out = []
    cardinality_out.append(1)
    for c in columns:
        cardinality.append(len(c.unique()))
    cardinality.pop()
    lenCol = len(columns)
    prod = 1
    for i in range(1, lenCol):
        prod *= cardinality[i - 1]
        cardinality_out.append(prod)

    cross_columns = cross_frame.apply(lambda row: prodm(row, cardinality_out), axis=1)

    return cross_columns


def one_hot(data):
    """
    Using pandas to convert the 'data' into a one_hot enconding format.
    """
    one_hot_table = pd.get_dummies(data.unique())
    one_hot = data.apply(lambda x: one_hot_table[x] == 1).astype(int)
    return one_hot


class CensusDataset(object):
    """
    To process the data the pandas library is being use.
    The features come from the census data.
    https://archive.ics.uci.edu/ml/datasets/census+income
    The data loader use 11 features (columns) for the deep stream.
    The data loader use 3 cross column for the wide section.
    """

    def __init__(self, batch_size):
        """
        The census data have two files. One for training and one for testing.
        """
        self.batch_size = batch_size

        self.COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
                        "marital_status", "occupation", "relationship", "race", "gender",
                        "capital_gain", "capital_loss", "hours_per_week", "native_country",
                        "income_bracket"]

        print("Loading training data ...")
        try:
            self.data = pd.read_csv("adult.data",
                                    names=self.COLUMNS,
                                    skipinitialspace=True, engine="python")
        except IOError as e:
            print("I/O error({0}): {1}. Donwload the Census 'adult.data' file from " +
                  "https://archive.ics.uci.edu/ml/datasets/adult. ".format(e.errno, e.strerror))
            sys.exit(1)
        except:
            print
            "Unexpected error:", sys.exc_info()[0]
            raise

        print("Loading validation data ...")
        try:
            self.test = pd.read_csv("adult.test",
                                    names=self.COLUMNS,
                                    skipinitialspace=True, engine="python", skiprows=1)
        except IOError as e:
            print("I/O error({0}): {1}. Donwload the Census 'adult.test' file from " +
                  "https://archive.ics.uci.edu/ml/datasets/adult. ".format(e.errno, e.strerror))
            sys.exit(1)
        except:
            print
            "Unexpected error:", sys.exc_info()[0]
            raise

        # Remove rows with empty records.

        print("Removing rows with empty records...")
        self.data = self.data.dropna(how="any", axis=0)
        self.test = self.test.dropna(how="any", axis=0)

        # Data preparation.

        print("Transforming the data to be ready for model training...")

        # Age will get discretize in 11 buckets.

        age_boundaries = [-1, 18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 120]
        age_labels = np.arange(len(age_boundaries) - 1)

        # We apply the process to the test and train data.

        self.age_buckets = pd.cut(self.data["age"], bins=age_boundaries, retbins=False,
                                  labels=age_labels)
        self.age_buckets_test = pd.cut(self.test["age"], bins=age_boundaries, retbins=False,
                                       labels=age_labels)

        # Using pandas to convert the text based categorial labels into numbers.
        # The data come in text form wich need to be converted in numerical form.
        # Each column get categorize, so every label will get a number associated with it.

        # Train

        self.gender_num_category = categorical_numbers(self.data["gender"])
        self.education_num_category = categorical_numbers(self.data["education"])
        self.marital_status_category = categorical_numbers(self.data["marital_status"])
        self.relationship_num_category = categorical_numbers(self.data["relationship"])
        self.workclass_num_category = categorical_numbers(self.data["workclass"])

        # Test

        self.gender_num_category_test = categorical_numbers(self.test["gender"])
        self.education_num_category_test = categorical_numbers(self.test["education"])
        self.marital_status_category_test = categorical_numbers(self.test["marital_status"])
        self.relationship_num_category_test = categorical_numbers(self.test["relationship"])
        self.workclass_num_category_test = categorical_numbers(self.test["workclass"])

        # Occupation and Native country will get hash and sparsify in 1000 size buckets
        # = hash(label)%1000.
        # Train

        self.occupation_hashed = self.data["occupation"].apply(
            lambda x: sparse_column_with_hash_bucket(x, 1000))
        self.native_country_hashed = \
            self.data["native_country"].apply(lambda x: sparse_column_with_hash_bucket(x, 1000))
        # Test

        self.occupation_hashed_test = \
            self.test["occupation"].apply(lambda x: sparse_column_with_hash_bucket(x, 1000))
        self.native_country_hashed_test = \
            self.test["native_country"].apply(lambda x: sparse_column_with_hash_bucket(x, 1000))

        # In the tutorial we will use two embedding layers.
        # Number of embedding layers

        self.embeddings = 2

        # The size of the embedding will be 8 for each column.

        self.embedding_dimension = 8

        # Count unique values for the index of the embedding layer.

        self.occupation_vocab = len(self.data["occupation"].unique())

        # Train

        self.occupation_embedding_index = categorical_numbers(self.data["occupation"])

        # Test

        self.occupation_embedding_index_test = categorical_numbers(self.test["occupation"])

        self.native_country_vocab = len(self.data["native_country"].unique())

        # Train

        self.native_country_embedding_index = categorical_numbers(self.data["native_country"])

        # Test

        self.native_country_embedding_index_test = \
            categorical_numbers(self.test["native_country"])

        # cross columns
        # Train
        # education x occupation

        self.occupation_num_category = categorical_numbers(self.data["occupation"])
        self.education_x_occupation = cross_columns(
            [self.occupation_num_category, self.education_num_category])
        self.education_x_occupation = \
            self.education_x_occupation.apply(lambda x: sparse_column_with_hash_bucket(x, 1000))

        # Test

        self.occupation_num_category_test = categorical_numbers(self.test["occupation"])
        self.education_x_occupation_test = cross_columns([self.education_num_category_test,
                                                          self.occupation_num_category_test])
        self.education_x_occupation_test = \
            self.education_x_occupation_test.apply(
                lambda x: sparse_column_with_hash_bucket(x, 1000))

        # Train
        # native country and occupation

        self.native_country_num_category = categorical_numbers(self.data["native_country"])
        self.native_country_x_occupation = cross_columns([self.occupation_num_category,
                                                          self.native_country_num_category])
        self.native_country_x_occupation = \
            self.native_country_x_occupation.apply(
                lambda x: sparse_column_with_hash_bucket(x, 1000))

        # Test

        self.native_country_num_category_test = categorical_numbers(self.test["native_country"])
        self.native_country_x_occupation_test = \
            cross_columns([self.occupation_num_category_test,
                           self.native_country_num_category_test])
        self.native_country_x_occupation_test = \
            self.native_country_x_occupation_test.apply(
                lambda x: sparse_column_with_hash_bucket(x, 1000))

        # Train
        # age_buckets, "education", "occupation"
        # cross columns
        # age buckets education and occupation

        self.age_buckets_education_occupation = cross_columns(
            [self.occupation_num_category, self.age_buckets,
             self.education_num_category])
        self.age_buckets_education_occupation = \
            self.age_buckets_education_occupation.apply(
                lambda x: sparse_column_with_hash_bucket(x, 1000))

        # Test

        self.age_buckets_education_occupation_test = cross_columns(
            [self.occupation_num_category_test,
             self.age_buckets_test,
             self.education_num_category_test])
        self.age_buckets_education_occupation_test = \
            self.age_buckets_education_occupation_test.apply(
                lambda x: sparse_column_with_hash_bucket(x, 1000))

        # Build the one hot encodings of the deep columns
        # Train

        self.workclass_one_hot = one_hot(self.data["workclass"])
        self.education_one_hot = one_hot(self.data["education"])
        self.gender_one_hot = one_hot(self.data["gender"])
        self.relationship_one_hot = one_hot(self.data["relationship"])

        # Test

        self.workclass_one_hot_test = one_hot(self.test["workclass"])
        self.education_one_hot_test = one_hot(self.test["education"])
        self.gender_one_hot_test = one_hot(self.test["gender"])
        self.relationship_one_hot_test = one_hot(self.test["relationship"])

        # Generating the labels.
        # This a binary classification problem. We convert the income bracket in two classes.
        # One above 50k income and one under.

        # Train

        self.labels = self.data["income_bracket"].apply(lambda x: ">50K" in x).astype(int)

        # Test

        self.labels_test = self.test["income_bracket"].apply(lambda x: ">50K" in x).astype(int)

        # Train

        self.deep_data = self.data[
            ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]]
        self.deep_frame = [self.deep_data]
        self.deep_data = pd.concat(self.deep_frame, axis=1)

        # Test

        self.deep_data_test = self.test[
            ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]]
        self.deep_frame_test = [self.deep_data_test]
        self.deep_data_test = pd.concat(self.deep_frame_test, axis=1)

        # Train
        # Linear / Wide Features for W & D model.

        self.frames = [self.education_x_occupation, self.native_country_x_occupation,
                       self.age_buckets_education_occupation]

        # Test

        self.frames_test = [self.education_x_occupation_test,
                            self.native_country_x_occupation_test,
                            self.age_buckets_education_occupation_test]

        # Train
        # Wide Data contains the index for the embeddings as well. +1 per embedding layer.

        self.wide_data = pd.concat(self.frames, axis=1)

        # Test

        self.wide_data_test = pd.concat(self.frames_test, axis=1)

        # The data need to be normalize becasue there is sigmoid activation function in the model.

        print("Normalizing the data...")
        # Train
        # Normalize wide data

        self.all_wide_data = self.wide_data.values
        self.normalizer = preprocessing.MinMaxScaler()
        self.scaled_data_wide = self.normalizer.fit_transform(self.all_wide_data)
        self.wide_data = pd.DataFrame(self.scaled_data_wide)

        # Test
        # Normalize wide test data

        self.all_wide_data_test = self.wide_data_test.values
        self.scaled_data_wide_test = self.normalizer.fit_transform(self.all_wide_data_test)
        self.wide_data_test = pd.DataFrame(self.scaled_data_wide_test)

        # Train
        # Normalize deep data

        self.all_data_deep = self.deep_data.values
        self.scaled_data_deep = self.normalizer.fit_transform(self.all_data_deep)
        self.deep_data = pd.DataFrame(self.scaled_data_deep)

        # Test
        # Normalize deep test data

        self.all_data_deep_test = self.deep_data_test.values
        self.scaled_data_deep_test = self.normalizer.fit_transform(self.all_data_deep_test)
        self.deep_data_test = pd.DataFrame(self.scaled_data_deep_test)

        # The occupation index table should NOT be normalize, becasue is an index to the
        #  embedding the model will learn.

        # The indicator column is already normalize by definition.
        # Train

        self.deep_frame_final = [self.deep_data, self.workclass_one_hot, self.education_one_hot,
                                 self.gender_one_hot,
                                 self.relationship_one_hot]
        self.deep_data = pd.concat(self.deep_frame_final, axis=1)

        self.embeddings_index = pd.concat([self.occupation_embedding_index.to_frame(),
                                           self.native_country_embedding_index.to_frame()],
                                          axis=1)

        # Test

        self.deep_frame_final_test = [self.deep_data_test, self.workclass_one_hot_test,
                                      self.education_one_hot_test,
                                      self.gender_one_hot_test, self.relationship_one_hot_test]
        self.deep_data_test = pd.concat(self.deep_frame_final_test, axis=1)

        self.embeddings_index_test = \
            pd.concat([self.occupation_embedding_index_test.to_frame(),
                       self.native_country_embedding_index_test.to_frame()], axis=1)

        print("Train Data Size:", self.data.values.shape)
        print("Train Labels Size:", self.labels.values.shape)

        print("Test Data Size:", self.test.values.shape)
        print("Test Labels Size:", self.labels_test.values.shape)

        # Number of samples in the training dataset.

        self.train_size = int(self.data.values.shape[0])

        # Number of samples in the testing dataset.

        self.test_size = int(self.test.values.shape[0])
        self.parameters = {}
        self.parameters['continuous_features'] = 5
        self.parameters['indicators_features'] = 33
        self.parameters['dimensions_embeddings'] = [8, 8]
        self.parameters['tokens_in_embeddings'] = []
        self.parameters['tokens_in_embeddings'].append(self.occupation_vocab)
        self.parameters['tokens_in_embeddings'].append(self.native_country_vocab)
        self.parameters['linear_features'] = 3

    def gendata(self, number_of_batches, deep_features, embeddings_features, wide_features,
                labels):

        size_deep = self.parameters['continuous_features'] + self.parameters['indicators_features']

        batch_number = 0
        sample_number = 0

        batch_x_wide = np.empty([self.batch_size, self.parameters['linear_features']])
        batch_x_embedding = np.empty([self.batch_size, self.embeddings])
        batch_x_deep = np.empty([self.batch_size, size_deep])
        batch_y = np.empty([self.batch_size])

        batches = {}

        for rows_deep, rows_embedding, rows_wide, rows_label in \
                zip(deep_features.iterrows(), embeddings_features.iterrows(),
                    wide_features.iterrows(),
                    labels.to_frame().iterrows()):

            # first column index / second column data.

            x_d_s = rows_deep[1]
            x_d_s = np.array(x_d_s)

            x_w_s = rows_wide[1]
            x_w_s = np.array(x_w_s)

            x_e_s = rows_embedding[1]
            x_e_s = np.array(x_e_s)

            l = rows_label[1]
            l = np.array(l)

            # just one label is needed.

            ys = l

            batch_x_deep[sample_number] = x_d_s
            batch_x_wide[sample_number] = x_w_s
            batch_x_embedding[sample_number] = x_e_s
            batch_y[sample_number] = ys

            sample_number += 1
            if sample_number == self.batch_size:
                sample_number = 0

                batches[batch_number] = tuple(
                    (batch_x_deep.T, batch_x_wide.T, batch_x_embedding.T, batch_y))
                batch_x_wide = np.empty([self.batch_size, self.parameters['linear_features']])
                batch_x_embedding = np.empty([self.batch_size, self.embeddings])
                batch_x_deep = np.empty([self.batch_size, size_deep])
                batch_y = np.empty([self.batch_size])

                batch_number += 1
            if batch_number == number_of_batches:
                break

        return batches
