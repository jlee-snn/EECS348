#Contributes: Daegon Kim, Joseph Lee

#imports class: MHelper from file: helper.py
from helper import MHelper
from decision_tree import MDecisionTree

import argparse
import collections
from pprint import pprint
import time
import statistics

# //////////////////////////[ DECISION TREE REL - PRUNING ]///////////////////////////

def attribute_route_to_tree( attribute_route, classification ):

    cur_attribute_route = attribute_route[:]
    if not attribute_route:
        return classification

    tree = { cur_attribute_route[0][0] : {} }
    attr = cur_attribute_route.pop(0)
    tree[ attr[0] ] = { attr[1] : attribute_route_to_tree( cur_attribute_route, classification ) }

    return tree

def accuracy_of_attribute_route( instances_validation, attribute_route, vec_Attributes ):
    # attribute_route ex. [ [ "attribute_a", attribute_a_val ], [ "attribute_b":attribute_b_val ] ]

    # Get the majority classification for current attribute route
    # Method: keep reducing instances_split through iteration
    instances_split = instances_validation[:]

    for attr in attribute_route:
        m_attr = next(a for a in vec_Attributes if a.attr_name == attr[0])
        instances_split = MHelper.split_instances( instances_split, m_attr )

        if m_attr.attr_type == 2: # nominal
            if attr[1] in instances_split.keys():
                instances_split = instances_split[ attr[1] ]
            else:
                instances_split = instances_validation
        else: # numeric
            ## [ NEED TO UPDATE ]
            try:
                attr_cur_min = float( attr[1][3:] )
            except ValueError:
                a = 3

            attr_max = max( [ float( a[3:] ) for a in instances_split.keys() ] )
            attr_min = min( [ float( a[3:] ) for a in instances_split.keys() ] )
            if attr_cur_min > attr_max:
                instances_split = instances_split[ ">= {}".format( attr_max ) ]
            elif attr_cur_min < attr_min:
                instances_split = instances_split[ ">= {}".format( attr_min ) ]
            else:
                instances_split = instances_split[ ">= {}".format( attr_cur_min ) ]
                asdf = 3

    majority_value = MHelper.majority_value_of_attribute( instances_split, -1 )

    # Create tree
    tree_route = attribute_route_to_tree( attribute_route, majority_value )

    #from pprint import pprint
    #pprint(tree_route)

    # Return accuracy of created tree
    return MDecisionTree.classification_accuracy(
        tree_route, instances_validation, vec_Attributes, -1 ), majority_value

def prune_tree( tree_trained, instances_validation, vec_Attributes,
                                 attribute_route=[], default_class=None ):

    # Do until further pruning is harmful:
    # 1. Evaluate impact on validation set of pruning each possible node (plus those below it)
    # 2. Greedily remove the one that most improves (validation set) accuracy

    # [RECURSION BASE CASES]
    if not tree_trained:
        return 0, 0.0
    # If tree_trained is not a dict, means that it has reached the bottom of the tree (class classification)
    if not isinstance(tree_trained, collections.OrderedDict):
        # return default_class
        return tree_trained, 1.0

    # [GO DOWN THE DECISION TREE]
    # each step of the decision tree only has 1 attribute assigned (hence idx: 0)
    # 1 attribute can have many values => go down the route of the value that's in the instance
    attribute_name = list(tree_trained.keys())[0]
    attribute_values = list(tree_trained.values())[0]

    # Get the classification if just used up until current attribute
    # Deep Copy? Shallow copy?
    attribute_route_cur = attribute_route[:]

    tree_pruned = { attribute_name: {} }
    curtree_accuracy = [ [ "", "", 0 ], -101 ]
    if len(attribute_route_cur) > 0:
        curtree_accuracy = accuracy_of_attribute_route(
            instances_validation, attribute_route_cur, vec_Attributes )

    for attr_val in attribute_values.keys():
        attribute_route_cur.append( [ attribute_name, attr_val ] )
        subtree_accuracy = accuracy_of_attribute_route(
            instances_validation, attribute_route_cur, vec_Attributes )

        # if subtree accuracy > curtree accuracy : add subtree to pruned tree
        # else: don't add
        if subtree_accuracy[0][2] > curtree_accuracy[0][2]:
            tree_pruned[attribute_name][attr_val] = prune_tree(
                attribute_values[attr_val], instances_validation, vec_Attributes, attribute_route_cur
            )[0]
        else:
            print( "subtree accuracy: {:.2f} <= curtree accuracy: {:.2f} => prune // attribute_route: {}".format(
                subtree_accuracy[0][2], curtree_accuracy[0][2], attribute_route_cur ) )
            tree_pruned[attribute_name][attr_val] = subtree_accuracy[1]

        attribute_route_cur.pop(-1)

    return tree_pruned, curtree_accuracy

# //////////////////////////[ DECISION TREE REL - ATTRIBUTES ]///////////////////////////

class MAttribute:
    def __init__(self, attr_idx, attr_name, attr_type, attr_range):
        self.attr_idx = attr_idx #attribute idx within instance (not idx within vec_Attributes)
        self.attr_name = attr_name
        self.attr_type = attr_type # 0: binary (output) / 1: numeric (input: continuous) / 2: nominal (input: discrete)
        self.attr_range = attr_range # only used for numeric values (ex. [start, end] : start <= x < end

def get_min_max_of_numeric_attribute( instances, attr_idx ):
        min = float(instances[0][attr_idx])
        max = float(instances[0][attr_idx])

        for instance in instances:
            if float(instance[ attr_idx ]) < min: min = float(instance[ attr_idx ])
            if float(instance[ attr_idx ]) > max: max = float(instance[ attr_idx ])

        return float(min), float(max)

def update_numeric_attributes( instances, vec_Attributes ):
    for attr in vec_Attributes:
        if attr.attr_type == 1:
            attr.attr_range[0], attr.attr_range[1] = get_min_max_of_numeric_attribute( instances, attr.attr_idx )

def fill_missing_attributes( instances ):

    print( "Filling in missing data...." )
    instance_ex = instances[0]
    attr_medians = [];
    for idx in range( len(instance_ex) ):
        attr_median = statistics.median_low([
                instance[idx] for instance in instances if instance[idx] != '?' ])
        attr_medians.append( attr_median )

    filled_instances = []

    for instance in instances:
        if '?' in instance:
            indices = [i for i, x in enumerate(instance) if x == "?"] # since multiple attributes could be missing
            for idx in indices:
                instance[idx] = attr_medians[ idx ]
        filled_instances.append( instance )

    print( "....complete. starting next step...")
    return filled_instances

# //////////////////////////[ DECISION TREE REL - QUESTION HELPERS  ]///////////////////////////

def count_splits( tree ):

    if not isinstance(tree, dict): #means hit the last leaf
        return 0

    number_of_splits = 0

    attr_subtrees = list(tree.values())[0]
    number_of_splits += len( attr_subtrees )

    for attr_name in attr_subtrees:
        number_of_splits += count_splits( attr_subtrees[ attr_name ] )

    return number_of_splits

def tree_to_disjunct_normal_boolean( tree, attribute_route=[], tree_disjunct_normal_form=[ "Tree = "], positive_count=0 ):

    #if len(attribute_route) >= 16: #only return at max, first 16 positive leaves
    #    return tree_disjunct_normal_form

    if len( tree_disjunct_normal_form ) >= 16:
        return

    if not isinstance(tree, dict): #means hit the last leaf
        if tree is "1":
            positive_count += 1
            tree_disjunct_normal_form_str = ""
            tree_disjunct_normal_form_str += " | (" if len( tree_disjunct_normal_form ) >= 2 else " ("
            for idx, name_val_pair in enumerate( attribute_route ):
                prefix = " ^ (" if idx > 0 else "("
                tree_disjunct_normal_form_str += prefix + " '{}' is '{}' )".format( name_val_pair[0], name_val_pair[1] )
            tree_disjunct_normal_form_str += ") \n"
            # [append]
            tree_disjunct_normal_form.append( tree_disjunct_normal_form_str )
        return

    attribute_route_cur = attribute_route[:] #shallow copy
    cur_attr_name = list(tree.keys())[0]
    attr_subtrees = list(tree.values())[0]

    for cur_attr_split_val in attr_subtrees:
        attribute_route_cur.append( [ cur_attr_name, cur_attr_split_val ] ) #append
        tree_to_disjunct_normal_boolean(
            attr_subtrees[ cur_attr_split_val ], attribute_route_cur,
            tree_disjunct_normal_form, positive_count ) #recurse
        attribute_route_cur.pop(-1) #pop

    return tree_disjunct_normal_form

# ///////////////////////////////[MAIN]////////////////////////////////
import csv

def main():

    g_print_tree = False
    g_prune_tree = False
    g_numeric_splits = 3
    g_learning_curve_splits = 10
    g_print_disjunct_normal_form = False

    g_file_train = ""
    g_file_validate = ""
    g_file_predict = ""

    # [COMMAND LINE ARGUMENTS]
    parser = argparse.ArgumentParser(description='Create / analyse a decision tree.')
    # optional analysis
    parser.add_argument('--prune', action='store_true',
                       help='prune the trained tree and use for validation')
    parser.add_argument('--print', action='store_true',
                       help='print the created trees (models)')
    parser.add_argument('--dnf', action='store_true',
                       help='print the boolean disjunct normal form of trees created')

    # analysis parameters (optional)
    parser.add_argument('--train', nargs='?', type=str, default="btrain-processed.csv",
                   help='create trained trees with the file specified (default: "btrain-processed.csv")')
    parser.add_argument('--validate', nargs='?', type=str, default="bvalidate.csv",
                   help='validate generated trees with the file specified (default: "bvalidate.csv"')
    parser.add_argument('--predict', nargs='?', type=str, default="btest.csv",
                   help='provide class predictions for the file specified (default: "btest.csv"')
    parser.add_argument('--learning', nargs='?', type=int, default=10,
                   help='specify the gradations for the learning curve (default: 10)')
    args = parser.parse_args()

    if args.prune:
        g_prune_tree = True
    if args.print:
        g_print_tree = True
    if args.dnf:
        g_print_disjunct_normal_form = True

    g_file_train = args.train
    g_file_validate = args.validate
    g_file_predict = args.predict
    g_learning_curve_splits = args.learning #default: 10

    g_prune_tree = True
    g_print_disjunct_normal_form = True
    #g_learning_curve_splits = 3

    # record exeuction time: start
    time_start = time.clock()

    # [ TODO ]
    # define cleaning method (how to deal with missing attributes ?)
    # clean up dealing with values outside range (numeric)
    # learning curve
    # Write up answers to questions

    # /////////////////////////////////////////////////////////////////////////
    # /////////////[ TRAIN - MAKE DECISION TREE ]//////////////
    # /////////////////////////////////////////////////////////////////////////
    vec_Attributes = []
    dic_ExamplesTrain = []
    input_filename = g_file_train
    print("Training Decision Tree on data. Please wait...\n")

    with open (input_filename, newline='') as csvfile_train:
        reader_train = csv.reader(csvfile_train, delimiter=' ', quotechar='|')

        # [READ IN EXAMPLE DATA: READ IN ONLY 'CLEAN' EXAMPLES (no missing data]
        for idx, row in enumerate(reader_train):
            cur_Example = ', '.join(row)
            if idx > 1: #and '?' not in cur_Example:
                dic_ExamplesTrain.append( cur_Example.split(",") )

        dic_ExamplesTrain = fill_missing_attributes( dic_ExamplesTrain )

        # [GET ATTRIBUTES AND ATTRIBUTE TYPES]
        # 0: binary (output) / 1: numeric (input: continuous) / 2: nominal (input: discrete)
        csvfile_train.seek(0)
        vec_AttributeNames = (', '.join(next(reader_train))).split(",,")
        vec_AttributeTypes = (', '.join(next(reader_train))).split(",,")
        #print( len(vec_AttributeNames)-1 )

        for idx in range(0, len(vec_AttributeNames)-1):
            if int( vec_AttributeTypes[idx] ) is 2:
                vec_Attributes.append( MAttribute(
                    idx, # 2: attr_idx
                    vec_AttributeNames[idx], # 0: attr_name
                    int( vec_AttributeTypes[idx] ), # 1: attr_type
                    []
                ) )
            else: #numeric (continuous)
                attr_min, attr_max = get_min_max_of_numeric_attribute( dic_ExamplesTrain, idx )
                attr_split = (attr_max - attr_min) / g_numeric_splits
                vec_Attributes.append( MAttribute(
                        idx, # 2: attr_idx
                        vec_AttributeNames[idx], # 0: attr_name
                        #"{} >= {}".format( vec_AttributeNames[idx], i * attr_split ), # 0: attr_name
                        int( vec_AttributeTypes[idx] ), # 1: attr_type
                        [ attr_min, attr_max, g_numeric_splits ] # [ attr_min, attr_max ]
                    ) )

        update_numeric_attributes( dic_ExamplesTrain, vec_Attributes )

    # [CREATE DECISION TREE]
    tree_trained = MDecisionTree.create_decision_tree(
        dic_ExamplesTrain,
        vec_Attributes,
        len( dic_ExamplesTrain[0] ) -1  )

    # [PRINT RESULT]
    print( "\nTRAIN. input file: {} // Tree trained with {} instances:".format(
        input_filename, len(dic_ExamplesTrain) ) )

    if g_print_tree is True:
        pprint(tree_trained)

    number_splits = count_splits( tree_trained )
    print( "number of splits: {}".format( number_splits ) )

    if g_print_disjunct_normal_form is True:
        disjunct_normal_form = tree_to_disjunct_normal_boolean( tree_trained, [], [ "Tree = "] )
        print( "disjunct normal form (displays up until first 16 positive leaves):\n {} "
               .format( "".join(disjunct_normal_form) ))

    # /////////////////////////////////////////////////////////////////////////
    # ///////[ VALIDATION - TEST TRAINED TREE / PRUNING ]///////
    # /////////////////////////////////////////////////////////////////////////

    dic_ExamplesValidate = []
    input_filename = g_file_validate

    # [GET VALIDATION INSTANCES]
    with open (input_filename, "r", newline='') as csvfile_validate:
        reader_validate = csv.reader(csvfile_validate, delimiter=' ', quotechar='|')

        for idx, row in enumerate(reader_validate):
            cur_Example = ', '.join(row)
            #if idx > 0:
            if idx > 0 and '?' not in cur_Example:
                dic_ExamplesValidate.append( cur_Example.split(",") )

    # important : otherwise numeric values split too much
    # update_numeric_attributes( dic_ExamplesTrain, vec_Attributes )

    validation_accuracy = MDecisionTree.classification_accuracy(
        tree_trained, dic_ExamplesValidate, vec_Attributes, -1  )

    # [PRINT RESULTS]
    print( "\nVALIDATION. input file: {} // accuracy: {}".format(
        input_filename, validation_accuracy[2]
    ) )

    # [PRUNING]
    if g_prune_tree:
        tree_pruned = prune_tree( tree_trained, dic_ExamplesValidate, vec_Attributes )[0]

        # print
        if g_print_tree is True:
           pprint(tree_pruned)

        if g_print_disjunct_normal_form is True:
            disjunct_normal_form = tree_to_disjunct_normal_boolean( tree_pruned, [], [ "Tree = "] )
            print( "disjunct normal form (displays up until first 16 positive leaves):\n {} "
                   .format( "".join(disjunct_normal_form) ))

        validation_accuracy_pruned = MDecisionTree.classification_accuracy(
            tree_pruned, dic_ExamplesValidate, vec_Attributes, -1  )

        print( "\nVALIDATION-pruned. input file: {} // accuracy: {}".format(
            input_filename, validation_accuracy_pruned[2]
        ) )

        number_splits = count_splits( tree_pruned )
        print( "number of splits: {}".format( number_splits ) )

    # [LEARNING CURVE]
    print("\nCalculating learning curve. Please wait....\n")

    # unpruned
    learning_curve = MDecisionTree.compute_learning_curve(
        tree_trained, dic_ExamplesValidate, vec_Attributes, g_learning_curve_splits )
    print( "learning curve (unpruned): \n{} ".format(
        "".join(
            [ "scale:{:.3f}x, accuracy:{} \n".format( accuracy[0], accuracy[1] ) for accuracy in learning_curve ]
        ) ) )

    # pruned
    if g_prune_tree:
        learning_curve = MDecisionTree.compute_learning_curve(
            tree_pruned, dic_ExamplesValidate, vec_Attributes, g_learning_curve_splits )
        print( "learning curve (pruned): \n{} ".format(
            "".join(
                [ "scale:{:.3f}x, accuracy:{} \n".format( accuracy[0], accuracy[1] ) for accuracy in learning_curve ]
            ) ) )

    # /////////////////////////////////////////////////////////////////////////
    # /////////////[ TEST - FILL WITH PREDICTIONS ]//////////////
    # /////////////////////////////////////////////////////////////////////////

    dic_ExamplesTest = []
    vec_AttributesFull = []
    print("\nOutputting predictions. Please wait....\n")

    # [GET TEST INSTANCES]
    with open (g_file_predict, "r", newline='') as csvfile_test:
        reader_test = csv.reader(csvfile_test, delimiter=' ', quotechar='|')

        for idx, row in enumerate(reader_test):
            cur_Example = ', '.join(row)
            if idx == 0:
                vec_AttributesFull = cur_Example.split(",")
                vec_AttributesFull = [attr for attr in vec_AttributesFull if attr != ""]
            if idx > 0:
                dic_ExamplesTest.append( cur_Example.split(",") )

    # This seems to take in dic_ExamplesTest as reference
    # test output: use pruned tree if exists, else unpruned tree
    MDecisionTree.predict_on_instances(
        tree_trained if g_prune_tree is False else tree_pruned, dic_ExamplesTest, vec_Attributes, -1
    )

    # [WRITE OUTPUT FILE]
    ouput_filename = "btest-predicted.csv"
    with open (ouput_filename, "w", newline='') as csvfile_test_predicted:
        writer_predicted = csv.writer(csvfile_test_predicted, delimiter=',', quotechar=' ')

        writer_predicted.writerow( vec_AttributesFull )
        for instance in dic_ExamplesTest:
            writer_predicted.writerow( instance )

    print( "\nPREDICTION. output file: {} // classified: {} instances.".format(
        ouput_filename, len( dic_ExamplesTest )
    ) )


    # /////////////////////////////////////////////////////////////////////////
    # /////////////////////////////////////////////////////////////////////////
    # /////////////////////////////////////////////////////////////////////////

    time_end = time.clock()
    print( "\nTOTAL EXECUTION TIME: {} ".format( time_end - time_start ))

# //////////////////////[ START ]//////////////////////
if __name__=="__main__":
   main()






