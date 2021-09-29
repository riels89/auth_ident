from auth_ident.models.contrastive_bilstm_v3 import ContrastiveBilstmV3
from auth_ident.models.contrastive_bilstm import ContrastiveBiLSTM
from auth_ident.models.contrastive_cnn import ContrastiveCNN
from auth_ident.models.generic_secondary_classifier import GenericSecondaryClassifier
from auth_ident.models.random_forest import RandomForestSecondaryClassifier
from auth_ident.models.k_neighbors import KNeighborSecondaryClassifier
from auth_ident.models.svm import SVMSecondaryClassifier
from auth_ident.models.cosknn import CosKNNSecondaryClassifier
from auth_ident.models.end_to_end_secondary import EndToEndMLP
#from auth_ident.models.histogram_verifier import HistogramVerifier

model_map = {
    "contrastive_bilstm_v3": ContrastiveBilstmV3,
    "contrastive_bilstm": ContrastiveBiLSTM,
    "contrastive_cnn": ContrastiveCNN,

    "random_forest": RandomForestSecondaryClassifier,
    "k_neighbors": KNeighborSecondaryClassifier,
    "svm": SVMSecondaryClassifier,
    "cosknn": CosKNNSecondaryClassifier,
#    "histogram_verifier": HistogramVerifier,
    "end_to_end_mlp": EndToEndMLP 
}
