#'@title LSTM com Normalização Adaptativa Integrada (pesos aprendidos)
#'@description Wrapper R para `ts_lstm_anw.py`. A normalização adaptativa é
#' parte da rede: uma média móvel ponderada é aprendida end-to-end junto com
#' os demais pesos. Mantém a interface do `ts_regsw`.
#'
#'@details Opções configuráveis via `obj$model` (PyTorch):
#' - `_aw_last_n`: inteiro. 0=todas as colunas; N=últimas N colunas na média.
#' - `_aw_temp`: float. Temperatura do softmax dos pesos.
#' - `_huber_beta`: float. Parâmetro do SmoothL1 (Huber) para robustez.
#' - `_grad_clip_enabled`: bool. Habilita gradient clipping.
#' - `_grad_clip_norm`: float. Norma usada no clipping.
#'
#'@param preprocess Normalização no R (recomenda-se `ts_norm_none()` para evitar duplicidade).
#'@param input_size Inteiro, número de lags (tamanho da janela).
#'@param epochs Inteiro, épocas máximas de treino.
#'@return Objeto `ts_lstm_anw`.
#'@importFrom tspredit ts_regsw
#'@import reticulate
#'@export
ts_lstm_anw <- function(preprocess = NA, input_size = NA, epochs = 10000L) {
  obj <- tspredit::ts_regsw(preprocess, input_size)
  obj$epochs <- epochs
  class(obj) <- append("ts_lstm_anw", class(obj))
  return(obj)
}

#'@importFrom tspredit do_fit
#'@exportS3Method do_fit ts_lstm_anw
do_fit.ts_lstm_anw <- function(obj, x, y) {
  # Carrega o backend com normalização integrada aprendível
  reticulate::source_python("ts_lstm_anw.py")

  if (is.null(obj$model))
    obj$model <- ts_lstm_create(obj$input_size, obj$input_size)

  df_train <- as.data.frame(x)
  df_train$t0 <- as.vector(y)

  obj$model <- ts_lstm_fit(obj$model, df_train, obj$epochs, 0.001)
  return(obj)
}

#'@importFrom tspredit do_predict
#'@exportS3Method do_predict ts_lstm_anw
do_predict.ts_lstm_anw <- function(obj, x) {
  reticulate::source_python("ts_lstm_anw.py")

  X_values <- as.data.frame(x)
  X_values$t0 <- 0

  prediction <- ts_lstm_predict(obj$model, X_values)
  return(prediction)
}
