from pydantic import BaseModel, Field


class ModelHyperparameters(BaseModel):
    ngram_lower_bound: int = Field(description="Minimum number of characters for an ngram in the vectorizer")
    ngram_upper_bound: int = Field(description="Maximum number of characters for an ngram in the vectorizer")
    max_tfidf_features: int = Field(description="Total number of ngrams allowed as features in TFIDF vectorizer")
    scale_features: bool = Field(description="Whether or not to scale the TDFIDF features")
    use_abstract: bool = Field(description="Whether or not to use abstract text when available")
    C: float = Field(description="SVM regularization parameter")


