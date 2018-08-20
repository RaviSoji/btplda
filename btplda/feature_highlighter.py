from .prediction_teacher import PredcitionTeacher

def highlight(D, plda_classifier, set_size=1):
    """ set_size is the number of features that can be used. """
    prediction, logp_prediction = plda_classifier.predict(D)
    target_model = plda_classifier.model

    u_model = target_model.transform(D, from_space='D', to_space='U_model')
    teacher = PredictionTeacher(u_model, target_model)

    ts = teacher.gen_teaching_sets(set_size)
    logps = teacher.calc_logp_likelihood(k=prediction, teaching_sets=ts)
    
    best_ts = ts[np.argmax(logps)]
    img_filter = teacher.build_filter(best_ts)
    highlighted = teacher.apply_filter(img, img_filter)

    return highlighted
