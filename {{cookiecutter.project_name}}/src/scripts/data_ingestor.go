package main

import (
	"io"
	"fmt"
	"os"
	"path/filepath"
	"github.com/gorilla/websocket"
)

func main() {
	files, err := filepath.Glob("data/raw/*.csv")
	if err != nil {
		fmt.Println("Failed to read folder:", err)
		return
	}

	conn, _, err := websocket.DefaultDialer.Dial("ws://localhost:1882/data_ingest", nil)
	if err != nil {
		fmt.Println("Failed to connect to WebSocket:", err)
		return
	}
	defer conn.Close()
	n:=1
	for _, file := range files {
		sendFile(file, conn)
		fmt.Println(n)
		n++
	}
}

func sendFile(file string, conn *websocket.Conn) {
	f, err := os.Open(file)
	if err != nil {
		fmt.Printf("Failed to open file %s: %v\n", file, err)
		return
	}
	defer f.Close()

	content, err := io.ReadAll(f)
	if err != nil {
		fmt.Printf("Failed to read all data from file %s: %v\n", file, err)
		return
	}
	err = conn.WriteMessage(websocket.TextMessage, content)
	if err != nil {
		fmt.Printf("Failed to send data from %s: %v\n", file, err)
		return
	}

	fmt.Printf("Sent file: %s\n", file)
}