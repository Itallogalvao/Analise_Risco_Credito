# Construindo um Modelo Preditivo para Análise de Risco


# Configurando o diretório de trabalho
# Coloque entre aspas o diretório de trabalho que você está usando no seu computador
# setwd("C:/RFundamentos/Mini-Projeto-2")
# getwd()

# Carregando o dataset em um dataframe
credit.df <- read.csv("credit_dataset.csv", header = TRUE, sep = ",")
head(credit.df)

## Convertendo as variáveis para o tipo fator (categórica)
to.factors <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(df[[variable]])
  }
  return(df)
}

## Normalização
scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center=T, scale=T)
  }
  return(df)
}

# Normalizando as variáveis
numeric.vars <- c("credit.duration.months", "age", "credit.amount")
credit.df <- scale.features(credit.df, numeric.vars)

# Variáveis do tipo fator
categorical.vars <- c('credit.rating', 'account.balance', 'previous.credit.payment.status',
                      'credit.purpose', 'savings', 'employment.duration', 'installment.rate',
                      'marital.status', 'guarantor', 'residence.duration', 'current.assets',
                      'other.credits', 'apartment.type', 'bank.credits', 'occupation', 
                      'dependents', 'telephone', 'foreign.worker')

credit.df <- to.factors(df = credit.df, variables = categorical.vars)

# Dividindo os dados em treino e teste - 60:40 ratio
library(caTools)

set.seed(101) 
amostra <- sample.split(credit.df, SplitRatio = 0.70)


train.data = subset(credit.df, amostra == TRUE)


test.data = subset(credit.df, amostra == FALSE)

# indexes <- sample(1:nrow(credit.df), size = 0.6 * nrow(credit.df))
# train.data <- credit.df[indexes,]
# test.data <- credit.df[-indexes,]

# Feature Selection com randomForest
library(randomForest) 

model <- randomForest(credit.rating ~ .,
                       data = train.data,
                       ntree = 100,
                       nodesize = 10,
                       importance = T)

# Visualizando os resultados
varImpPlot(model)

# Criando e Avaliando o Modelo
library(caret) 
library(ROCR) 

# Biblioteca de utilitários para construção de gráficos
source("plot_utils.R") 

## separate feature and class variables
test.feature.vars <- test.data[,-1]
test.class.var <- test.data[,1]

# Construindo um modelo de regressão logística
lr.model <- glm(credit.rating ~ ., data = train.data, family = "binomial")

# Visualizando o modelo
summary(lr.model)
# Testando o modelo nos dados de teste
lr.predictions <- predict(lr.model, test.data, type="response")
lr.predictions <- round(lr.predictions)

# Avaliando o modelo
confusionMatrix(table(data = lr.predictions, reference = test.class.var), positive = '1')

## Feature selection
formula <- "credit.rating ~ ."
formula <- as.formula(formula)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 2)
model <- train(formula, data = train.data, method = "glm", trControl = control)
importance <- varImp(model, scale = FALSE)
plot(importance)


# Construindo o modelo com as variáveis selecionadas
formula.new <- "credit.rating ~ account.balance + credit.purpose + previous.credit.payment.status + savings + credit.duration.months"
formula.new <- as.formula(formula.new)
lr.model.new <- glm(formula = formula.new, data = train.data, family = "binomial")

# Visualizando o modelo
summary(lr.model.new)

# Testando o modelo nos dados de teste
lr.predictions.new <- predict(lr.model.new, test.data, type = "response") 
lr.predictions.new <- round(lr.predictions.new)

# Avaliando o modelo
confusionMatrix(table(data = lr.predictions.new, reference = test.class.var), positive = '1')

# Avaliando a performance do modelo

# Criando curvas ROC
lr.model.best <- lr.model
lr.prediction.values <- predict(lr.model.best, test.feature.vars, type = "response")
predictions <- prediction(lr.prediction.values, test.class.var)
par(mfrow = c(1,2))
plot.roc.curve(predictions, title.text = "Curva ROC")
plot.pr.curve(predictions, title.text = "Curva Precision/Recall")


# ~~~~~~~~~~~~~~ Modelo Naive Bayes ~~~~~~~~~~~~
library(e1071) 
lr.model.new <- naiveBayes(formula = formula.new, data = train.data)


# Visualizando o modelo
summary(lr.model.new)

y_pred <- predict(lr.model.new, newdata = test.data) 

# Confusion Matrix 
cm <- table(test.data$credit.rating, y_pred) 
cm 

# Model Evauation 
confusionMatrix(cm)

# ~~~~~~~~~~ Modelo SVM ~~~~~~~~~~~~~~~~

lr.model.new <- svm(formula = formula.new, data = train.data, 
                    type = 'C-classification', 
                    kernel = 'radial')

# Previsões nos dados de treino
pred_train <- predict(lr.model.new, train.data) 

# Percentual de previsões corretas com dataset de treino
mean(pred_train == train.data$credit.rating)  


# Previsões nos dados de teste
pred_test <- predict(lr.model.new, test.data) 

# Percentual de previsões corretas com dataset de teste
mean(pred_test == test.data$credit.rating)  

# Confusion Matrix
table(pred_test, test.data$credit.rating)



