from auth_ident.models.cnn_lstm import CNNLSTM
from auth_ident.models.contrastive_1D_to_2D import Contrastive1DTo2D
from auth_ident.models.contrastive_bilstm_v2 import ContrastiveBilstmV2
from auth_ident.models.contrastive_bilstm import ContrastiveBiLSTM
from auth_ident.models.contrastive_by_line_cnn import ContrastiveByLineCNN
from auth_ident.models.contrastive_cnn import ContrastiveCNN
from auth_ident.models.contrastive_stacked_bilstm import ContrastiveStackedBiLSTM
from auth_ident.models.dilated_conv_by_line import DilatedCNNByLine
from auth_ident.models.largeNN import LargeNN
from auth_ident.models.multi_attention_bilstm import MultiHeadAttentionBiLSTM
from auth_ident.models.simple_lstm import SimpleLSTM
from auth_ident.models.split_bilstm import SplitBilstm
from auth_ident.models.split_cnn import SplitCNN
from auth_ident.models.split_lstm import SplitLSTM
from auth_ident.models.split_NN import SplitNN
from auth_ident.models.large_contrastive_cnn import LargeContrastiveCNN
from auth_ident.models.generic_secondary_classifier import GenericSecondaryClassifier
from auth_ident.models.random_forest import RandomForestSecondaryClassifier
from auth_ident.models.k_neighbors import KNeighborSecondaryClassifier
from auth_ident.models.svm import SVMSecondaryClassifier
from auth_ident.models.universal_transformer import UniversalTransformer
from auth_ident.models.transformer import Transformer
from auth_ident.models.cosknn import CosKNNSecondaryClassifier
from auth_ident.models.end_to_end_secondary import EndToEndMLP

model_map = {
    "cnn_lstm": CNNLSTM,
    "contrastive_1D_to_2D": Contrastive1DTo2D,
    "contrastive_bilstm_v2": ContrastiveBilstmV2,
    "contrastive_bilstm": ContrastiveStackedBiLSTM,
    "contrastive_by_line_cnn": ContrastiveByLineCNN,
    "contrastive_cnn": ContrastiveCNN,
    "contrastive_stacked_bilstm": ContrastiveStackedBiLSTM,
    "dilated_conv_by_line": DilatedCNNByLine,
    "largeNN": LargeNN,
    "multi_attention_bilstm": MultiHeadAttentionBiLSTM,
    "simple_lstm": SimpleLSTM,
    "split_bilstm": SplitBilstm,
    "split_cnn": SplitCNN,
    "split_lstm": SplitLSTM,
    "split_NN": SplitNN,
    "large_contrastive_cnn": LargeContrastiveCNN,
    "universal_transformer": UniversalTransformer,
    "transformer": Transformer,

    "random_forest": RandomForestSecondaryClassifier,
    "k_neighbors": KNeighborSecondaryClassifier,
    "svm": SVMSecondaryClassifier,
    "cosknn": CosKNNSecondaryClassifier,
    "end_to_end_mlp": EndToEndMLP 
}
