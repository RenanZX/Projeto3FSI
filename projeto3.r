#SVM
library(e1071)
library(kernlab)
library(ggplot2)

#codigos de processamento foram baseados nesses artigos
#https://2engenheiros.com/2017/07/24/classificar-dados-svm-no-r/
#https://www.r-bloggers.com/machine-learning-using-support-vector-machines/
#https://www4.stat.ncsu.edu/~reich/BigData/code/SVM.html

leitor <- list(
  readfile = function(){
    data1 <- read.table(file.choose(),sep=',', header = TRUE)
    colnames(data1) <- c("IDnro","Diagnosis","radius","texture","perimeter","area",
                         "smoothness","compactness","concavity","concave points","symmetry","fractal dimension")
    as.data.frame(data1[,c(1,2,3,4,5,6,7,8,9,10,11,12,13)])
  }
)

processor <- list(
  ProcessarSVMKLinear = function(treino, teste){
    # Separando dados levantados das classificações
    error_svm <- 0:20
    
    for(i in 1:21){
      resultados <- teste$Diagnosis
      
      # Criando modelo SVM a partir do conjunto de dados
      modelo_svm <- svm(Diagnosis ~ ., data = treino, kernel="linear",type="C-classification")
      predteste <- predict(modelo_svm, teste)
      error_svm[i-1] <- ((1-sum(predteste == resultados)/length(predteste))*100)
    }
    
    C_value <- -5:15
    
    n <- max(length(error_svm),length(C_value))
    length(error_svm) <- n
    print(plot(C_value,error_svm,xlab = "Valores de C[2^-5,2^15]", ylab = "Erros de Teste(%)", main = "Erros de Teste x Valores de C",type="o",col="blue",axis=FALSE))
    axis(1,at=C_value,labels = 2^(C_value))
    TRUE
  },
  ProcessarSVMKGauss = function(treino, teste){
    error_svm <- 0:20
    
    for(i in 1:21){
      x <- sample(1:3,1)
      resultados <- teste$Diagnosis
        
      # Criando modelo SVM a partir do conjunto de dados
      modelo_gauss <- ksvm(Diagnosis ~., data=treino,kernel="rbfdot",epsilon=1)
      predteste <- predict(modelo_gauss, teste)
      error_svm[i-1] <- ((1-sum(predteste == resultados)/length(predteste))*100)
    }
    # Resumo do modelo
    #summary(modelo_svm)
    
    C_value <- -5:15
    
    n <- max(length(error_svm),length(C_value))
    length(error_svm) <- n
    print(plot(C_value,error_svm,xlab = "Valores de C[2^-5,2^15]", ylab = "Erros de Teste(%)", main = "Erros de Teste x Valores de C",type="o",col="blue",axis=FALSE))
    axis(1,at=C_value,labels = 2^(C_value))
    TRUE
  },
  ProcessarSVMKFBR = function(treino, teste){
    tune_svm <- tune(svm, Diagnosis ~ ., data = treino, kernel = "radial")
    error_svm <- 0:20
    
    for(i in 1:21){
      x <- sample(1:3,1)
      
      resultados <- teste$Diagnosis
      
      # Criando modelo SVM a partir do conjunto de dados
      modelo_svm_tuned <- svm(Diagnosis ~ ., treino, kernel = "radial",type="C-classification")
      
      predteste <- predict(modelo_svm_tuned, teste)
      error_svm[i-1] <- ((1-sum(predteste == resultados)/length(predteste))*100)
    }
    # Resumo do modelo
    #summary(modelo_svm)
    
    C_value <- -5:15
    
    n <- max(length(error_svm),length(C_value))
    length(error_svm) <- n
    print(plot(C_value,error_svm,xlab = "Valores de C[2^-5,2^15]", ylab = "Erros de Teste(%)", main = "Erros de Teste x Valores de C",type="o",col="blue",axis=FALSE))
    axis(1,at=C_value,labels = 2^(C_value))
    TRUE
  }
)

data = leitor$readfile()

train <- sample(nrow(data), 0.7*nrow(data), replace = FALSE)
TrainSet <- data[train,]
ValidSet <- data[-train,]

processor$ProcessarSVMKLinear(TrainSet,ValidSet)