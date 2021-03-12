from GraphicManager import GraphicManager
from GestuReNN import GestuReNN
from DataLoader import DataLoader

# Train/Test control panel
train_st_s = False
train_st_l = False
train_mt_s = False
train_mt_m = False

# Mutable variable setting
dataset = '1$'
load_mode = 'test'

if train_st_s or train_st_l or train_mt_s or train_mt_m:
    print('\n--------------------------------')
    print('You have selected training mode:')
    print('Train ST-S: {} - Train ST-L: {} - Train MT-S: {} - Train MT-M: {}'.format(train_st_s,
                                                                                     train_st_l,
                                                                                     train_mt_s,
                                                                                     train_mt_m))
    print('Stroke dataset {} with data loading mode {}'.format(dataset, load_mode))
    user_input = input('Are you sure you want to go on? Checkpoints will be overwritten (y/n): ')

    if user_input is not 'y':
        exit(1)

# Data loading
dl = DataLoader(dataset=dataset, load_mode=load_mode)
model_1 = GestuReNN(dataset=dataset, plot=False, topology='sts')
model_2 = GestuReNN(dataset=dataset, plot=False, topology='stl')
model_3 = GestuReNN(dataset=dataset, plot=False, topology='mts')
model_4 = GestuReNN(dataset=dataset, plot=False, topology='mtm')

# Setting Topology 1
if train_st_s:
    model_1.fit_clf(dl.train_set_classifier, dl.validation_set_classifier)
    model_1.fit_reg(dl.train_set_regressor, dl.validation_set_regressor)
else:
    model_1.load_model()

# Setting Topology 2
if train_st_l:
    model_2.fit_clf(dl.train_set_classifier, dl.validation_set_classifier)
    model_2.fit_reg(dl.train_set_regressor, dl.validation_set_regressor)
else:
    model_2.load_model()

# Setting Topology 3
if train_mt_s:
    model_3.fit_model(dl.train_set_classifier,
                      dl.validation_set_classifier,
                      dl.train_set_regressor,
                      dl.validation_set_regressor)
else:
    model_3.load_model()

# Setting Topology 4
if train_mt_m:
    model_4.fit_model(dl.train_set_classifier,
                      dl.validation_set_classifier,
                      dl.train_set_regressor,
                      dl.validation_set_regressor)
else:
    model_4.load_model()

models = [model_1, model_2, model_3, model_4]
graphic_manager = GraphicManager(dataset=dataset)
graphic_manager.compare_models(models, dl.test_set_classifier, best_of=1, mode='clf')
graphic_manager.compare_models(models, dl.test_set_classifier, best_of=3, mode='clf')
graphic_manager.compare_models(models, dl.test_set_classifier, best_of=5, mode='clf')
graphic_manager.compare_models(models, dl.test_set_classifier, mode='reg')

print('\n MSE for regressor \n')
graphic_manager.generate_step_accuracy(model_1, dl.test_set_classifier, steps=20, mode='reg')
graphic_manager.generate_step_accuracy(model_2, dl.test_set_classifier, steps=20, mode='reg')
graphic_manager.generate_step_accuracy(model_3, dl.test_set_classifier, steps=20, mode='reg')
graphic_manager.generate_step_accuracy(model_4, dl.test_set_classifier, steps=20, mode='reg')

print('\n Accuracy for model 1\n')
graphic_manager.generate_step_accuracy(model_1, dl.test_set_classifier, best_of=1, steps=20)
graphic_manager.generate_step_accuracy(model_1, dl.test_set_classifier, best_of=3, steps=20)
graphic_manager.generate_step_accuracy(model_1, dl.test_set_classifier, best_of=5, steps=20)

print('\n Accuracy for model 2\n')
graphic_manager.generate_step_accuracy(model_2, dl.test_set_classifier, best_of=1, steps=20)
graphic_manager.generate_step_accuracy(model_2, dl.test_set_classifier, best_of=3, steps=20)
graphic_manager.generate_step_accuracy(model_2, dl.test_set_classifier, best_of=5, steps=20)

print('\n Accuracy for model 3\n')
graphic_manager.generate_step_accuracy(model_3, dl.test_set_classifier, best_of=1, steps=20)
graphic_manager.generate_step_accuracy(model_3, dl.test_set_classifier, best_of=3, steps=20)
graphic_manager.generate_step_accuracy(model_3, dl.test_set_classifier, best_of=5, steps=20)

print('\n Accuracy for model 4\n')
graphic_manager.generate_step_accuracy(model_4, dl.test_set_classifier, best_of=1, steps=20)
graphic_manager.generate_step_accuracy(model_4, dl.test_set_classifier, best_of=3, steps=20)
graphic_manager.generate_step_accuracy(model_4, dl.test_set_classifier, best_of=5, steps=20)



