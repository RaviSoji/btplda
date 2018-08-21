import numpy as np

from .prediction_teacher import PredictionTeacher


def highlight(D, plda_classifier, set_size=1, return_filter=False):
    """ set_size is the number of features that can be used. """
    prediction, logp_prediction = plda_classifier.predict(D)
    target_model = plda_classifier.model

    u_model = target_model.transform(D, from_space='D', to_space='U_model')
    u_model = np.squeeze(u_model)

    teacher = PredictionTeacher(u_model, target_model)

    ts = teacher.gen_teaching_sets(set_size)
    logps = teacher.calc_logp_likelihood(k=prediction, teaching_sets=ts)
    
    best_ts = ts[np.argmax(logps)]
    data_filter = teacher.build_filter(best_ts)
    highlighted = teacher.apply_filter(D, data_filter)

    if return_filter:
        return highlighted, data_filter

    else:
        return highlighted
