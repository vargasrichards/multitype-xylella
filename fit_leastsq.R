# A. Vargas Richards, 29 Jan 2025
# fits the Cauchy kernel to the exponential kernel in R2


exp_fitted <- function(x){ # normalised
  return ((x/119)*exp(-x/119))
}

exp_fitted2 <- function(x){ # unnormalised
  return (exp(-x/119))
}

cauchyf <- function(x, alpha) { # cauchy
  return(1/(1 + (x/alpha)^2))
}

fitls <- function() {
  
  xseq <- seq(0, 8000, by = 0.1) # fit out to the order of ~ R*landscape length
  yseq <- exp_fitted2(xseq)  # fitting against unnormalised exponential kernel
  
  alpha_range <- seq(1, 250, by = .1) # 
  
  errors <- numeric(length(alpha_range))
  
  for (i in seq_along(alpha_range)) {
    alpha <- alpha_range[i]
    
    y_pred <- cauchyf(xseq, alpha)
    
    error <- sum((yseq - y_pred)^2)
    
    errors[i] <- error
  }
  
  plot(alpha_range, errors, type = "l", col = "green", 
       xlab = "Alpha", ylab = "SSE", 
       main = "Least Squares Fitting: Error vs. Alpha")
  
  optimal_alpha <- alpha_range[which.min(errors)]
  cat("Optimal alpha:", optimal_alpha, "\n")
  y_fitted <- cauchyf(xseq, optimal_alpha)
plot(xseq, yseq, col = "green", pch = 10, xlab = "Distance /m", ylab = "Density", 
       main = "Kernel Fitting: Exponential, Cauchy")
  lines(xseq, y_fitted, col = "pink", lwd = 2)  
  legend("topright", legend = c("Unnormalised Exponential, a = 119m ", "Fitted Cauchy (Unnormalised), a=84.5m"), 
         col = c("green", "pink"), pch = c(16, NA), lwd = c(1, 2))
  
  return(list(optimal_alpha = optimal_alpha, min_error = min(errors)))
}

fitls()

