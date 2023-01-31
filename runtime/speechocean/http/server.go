package main

import (
	"fmt"
	"net/http"

	"github.com/gin-gonic/gin"
)

/*
#cgo CFLAGS:
#cgo LDFLAGS: -lmodel
#include "model.h"
*/
import "C"

type DecodeResult struct {
	transcript string
	duration   int
	decodeTime int
}

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

	// 调用底层C++推理函数
	var wav_path string = "zh-cn-demo.wav"

	// 声明C语言返回类型，否则报错
	// cannot use (_Cfunc_predict)((_Cfunc_CString)(wav_path)) (value of type _Ctype_struct___0) as type DecodeResult in assignment
	var result C.CDecodeResult

	// 需要将go语言字符串类型转换成C语言的字符串类型，否则报错。
	// cannot use wav_path (variable of type string) as type *_Ctype_char in argument to (_Cfunc_predict)
	result = C.predict(C.CString(wav_path))

	// C.GoString()将C语言字符串转换成Go语言的字符串。
	fmt.Println("predict result:", C.GoString(result.transcript))

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
