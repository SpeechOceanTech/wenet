package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

/*
#cgo CFLAGS: -I libwenet
#cgo LDFLAGS: -lwenet
#include "model.h"
*/
import "C"

var tmpFileDir string = "/tmp/model-server"

type ModelInput map[string]string

type ModelPredictRequest struct {
	Parameters map[string]interface{} `json:parameters`
	Inputs     []ModelInput           `json:inputs`
}

type ModelPredictRespnose struct {
}

func Download(uri string) (string, error) {
	uid := uuid.New().String()
	// now := time.Now()

	u, err := url.Parse(uri)
	if err != nil {
		return "", err
	}

	pathList := strings.Split(u.Path, "/")
	filename := pathList[len(pathList)-1]

	// 解析URL
	var outputPath string = tmpFileDir + "/" + uid + filename
	fmt.Println("outputPath: ", outputPath)

	// 网络请求
	resp, err := http.Get(uri)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	// 写文件
	f, err := os.OpenFile(outputPath, os.O_CREATE|os.O_RDWR, 0666)
	if err != nil {
		return "", err
	}
	defer f.Close()

	_, err = io.Copy(f, resp.Body)
	if err != nil {
		return "", err
	}

	return outputPath, nil
}

func healthHandler(c *gin.Context) {
	resp := map[string]string{
		"status": "alive",
	}
	c.JSON(http.StatusOK, resp)
}

func statusHandler(c *gin.Context) {
	resp := map[string]interface{}{
		"name":  "model_name",
		"ready": true,
	}
	c.JSON(http.StatusOK, resp)
}

func listHandler(c *gin.Context) {
	var modelList []string
	modelList = append(modelList, "model_name")

	resp := map[string]interface{}{
		"models": modelList,
	}
	c.JSON(http.StatusOK, resp)
}

func predictHandler(c *gin.Context) {
	var request ModelPredictRequest
	resp := map[string]interface{}{
		"model_name":    "asr",
		"model_version": "1.0",
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		resp["error"] = err.Error()
		c.JSON(400, resp)
		return
	}

	var local_wav_path string
	var err error

	wav_path := request.Inputs[0]["wav_path"]
	if strings.HasPrefix(wav_path, "http") {
		local_wav_path, err = Download(wav_path)
		if err != nil {
			resp["error"] = "Download error, " + err.Error()
			c.JSON(http.StatusOK, resp)
			return
		}
		fmt.Println("Download", wav_path, " to ", local_wav_path)
	} else {
		local_wav_path = wav_path
	}

	// 调用底层C++推理函数
	// var wav_path string = "zh-cn-demo.wav"

	var modelRequest C.ModelRequest
	var modelResponse C.ModelResponse

	modelRequest.wav_path = C.CString(local_wav_path)

	// 需要将go语言字符串类型转换成C语言的字符串类型，否则报错。
	// cannot use wav_path (variable of type string) as type *_Ctype_char in argument to (_Cfunc_predict)
	modelResponse = C.model_predict(modelRequest)

	// C.GoString()将C语言字符串转换成Go语言的字符串。
	fmt.Println("predict result:", C.GoString(modelResponse.text))
	resp["text"] = C.GoString(modelResponse.text)

	c.JSON(http.StatusOK, resp)
}

func main() {
	fmt.Println("Running model server")

	// 创建临时文件目录
	if _, err := os.Stat(tmpFileDir); err != nil {
		if os.IsNotExist(err) {
			if err := os.Mkdir(tmpFileDir, os.ModePerm); err != nil {
				fmt.Println("Create directory error.")
				return
			}
		}
	}

	modelName := flag.String("model_name", "wenet", "model name")
	modelVersion := flag.String("model_version", "1.0", "model version")
	modelPath := flag.String("model_path", "", "model path")

	flag.Parse()

	// 检查模型目录是否存在
	if _, err := os.Stat(*modelPath); err != nil {
		if os.IsNotExist(err) {
			log.Fatal("model path not exist!")
			return
		}
	}

	// 调用底层C++代码，初始化、并加载模型
	C.model_load(C.CString(*modelName), C.CString(*modelVersion), C.CString(*modelPath))

	router := gin.Default()

	router.GET("/", healthHandler)
	router.GET("/health", healthHandler)
	router.GET("/models", listHandler)
	router.GET("/status", statusHandler)
	router.POST("/predict", predictHandler)
	router.POST("/infer", predictHandler)

	router.Run("0.0.0.0:8080")
}
