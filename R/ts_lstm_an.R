#'@title LSTM (Normalização Adaptativa Interna)
#'@description Previsor de séries com LSTM (PyTorch) realizando normalização
#' adaptativa dentro do backend Python. Mantém a interface de `ts_regsw`.
#' Recomenda-se usar `ts_norm_none()` no R, já que o Python trata a escala.
#'
#'@details Opções configuráveis via atributos no modelo Python (`obj$model`):
#' - `_an_nw`: inteiro. 0=usar todas as colunas; N=últimas N colunas na média.
#' - `_robust_clip`: bool. Ativa clipping por quantis antes da escala.
#' - `_clip_q_low`/`_clip_q_high`: limites de quantis para clipping (ex.: 0.01/0.99).
#' - `_robust_minmax`: bool. Usa min–max robusto por quantis.
#' - `_mm_q_low`/`_mm_q_high`: quantis para min–max robusto (ex.: 0.05/0.95).
#'
#'@param preprocess Opcional. Objeto de pré-processamento (recomendado `ts_norm_none()`).
#'@param input_size Inteiro. Número de lags por exemplo (tamanho da janela).
#'@param epochs Inteiro. Número máximo de épocas de treino.
#'@return Objeto `ts_lstm_an`.
#'
#'@examples
#'# Exemplo (sem normalização prévia no R):
#'# library(daltoolbox)
#'# library(reticulate)
#'# reticulate::source_python("ts_lstm_an.py")
#'# data(tsd)
#'# ts <- ts_data(tsd$y, 10)
#'# samp <- ts_sample(ts, test_size = 5)
#'# io_train <- ts_projection(samp$train)
#'# model <- ts_lstm_an(ts_norm_none(), input_size=4, epochs=2000L)
#'# # Opcional: ajustar parâmetros robustos antes do fit
#'# model$model$`_an_nw` <- 3
#'# model$model$`_robust_clip` <- TRUE
#'# model$model$`_robust_minmax` <- TRUE
#'# model <- fit(model, x=io_train$input, y=io_train$output)
#'@importFrom tspredit ts_regsw
#'@import reticulate
#'@export
ts_lstm_an <- function(preprocess = NA, input_size = NA, epochs = 10000L) {
  obj <- tspredit::ts_regsw(preprocess, input_size)
  obj$epochs <- epochs
  class(obj) <- append("ts_lstm_an", class(obj))
  return(obj)
}

#'@importFrom tspredit do_fit
#'@exportS3Method do_fit ts_lstm_an
do_fit.ts_lstm_an <- function(obj, x, y) {
  # Garante o carregamento do backend Python adaptativo local
  reticulate::source_python("ts_lstm_an.py")

  # Cria o modelo Python caso ainda não exista
  if (is.null(obj$model))
    obj$model <- ts_lstm_create(obj$input_size, obj$input_size)

  # Monta o data.frame de treino com a coluna alvo 't0'
  df_train <- as.data.frame(x)
  df_train$t0 <- as.vector(y)

  # Treina via Python (normalização adaptativa acontece lá)
  obj$model <- ts_lstm_fit(obj$model, df_train, obj$epochs, 0.001)
  return(obj)
}


#'@importFrom tspredit do_predict
#'@exportS3Method do_predict ts_lstm_an
do_predict.ts_lstm_an <- function(obj, x) {
  # Garante o carregamento do backend Python adaptativo local
  reticulate::source_python("ts_lstm_an.py")

  # Monta o data.frame de predição (coluna 't0' pode ser dummy/ignorada)
  X_values <- as.data.frame(x)
  X_values$t0 <- 0

  prediction <- ts_lstm_predict(obj$model, X_values)
  return(prediction)
}
