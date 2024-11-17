# Motor CC com velocidade controlada por RNA

# Modelo RNA de Regressão
- 5 camadas e 25 épocas
- Otimizador Adam
- Learning Rate 0.001
- BatchSize 32

<p align='left'>
  <img src='https://github.com/user-attachments/assets/e77b6064-a2c2-4cf7-a857-4563d5bde12a' width='750'>  
</p>

## Atributos
- Coluna 1 – “T” = Torque de Entrada
- Coluna 2 – “w” = Velocidade Angular de Entrada
- Coluna 3 – “i” = Corrente de Entrada
- Coluna 4 – “V” = Tensão que o Controle Fuzzy indicou como saída

## MAE
  Resultado de um MAE = 0.424380 
  
## Exemplos de Controle
<img width="369" alt="Figure 2024-11-16 222234" src="https://github.com/user-attachments/assets/946a2058-cbf1-4640-8dfe-eda5a6b1cf7e">

<img width="375" alt="Figure 2024-11-16 222227" src="https://github.com/user-attachments/assets/f5463768-6c81-4cd2-a65c-5a07b2792618">
