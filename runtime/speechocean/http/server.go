package main

import (
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"
)

/*
	#include "model.h"
*/
import "C"

func healthHandler(c *gin.Context) {
	resp := map[string]string{
		"status": "alive",
	}
	c.IndentedJSON(http.StatusOK, resp)
}

func statusHandler(c *gin.Context) {
	resp := map[string]interface{}{
		"name":  "model_name",
		"ready": true,
	}
	c.IndentedJSON(http.StatusOK, resp)
}

func listHandler(c *gin.Context) {
	var modelList []string
	modelList = append(modelList, "model_name")

	resp := map[string]interface{}{
		"models": modelList,
	}
	c.IndentedJSON(http.StatusOK, resp)
}

func predictHandler(c *gin.Context) {
	resp := map[string]interface{}{
		"model_name":    "asr",
		"model_version": "1.0",
	}

	// data := map[string]string{}

	// 调用C++模型推理函数
	predictResult := C.predict()
	fmt.Println("predict return:", predictResult)

	c.IndentedJSON(http.StatusOK, resp)
}

func main() {
	fmt.Println("Running model server")

	// 调用底层C++代码，初始化、并加载模型
	C.init()
	C.load()

	router := gin.Default()

	router.GET("/", healthHandler)
	router.GET("/health", healthHandler)
	router.GET("/models", listHandler)
	router.GET("/status", statusHandler)
	router.POST("/predict", predictHandler)
	router.POST("/infer", predictHandler)

	router.Run("0.0.0.0:8080")
}
