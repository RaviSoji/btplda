class Explainer:
    def __init__(self, fitted_plda_classifier):
        self.model = 


    def explain_category(self, category, n_ts=5, ts_sz=1):
        assert category in self.model.get_categories

        teaching_sets = 

        highlighter = FeatureHighlighter(
        highlighted, filters = 

        return teaching_sets, highlighted_teaching_sets
