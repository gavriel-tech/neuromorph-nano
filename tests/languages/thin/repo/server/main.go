package main

import (
	"fmt"
	"net/http"
)

// MaxBody is the largest request body the server reads, in bytes.
const MaxBody = 1 << 20

type Store struct {
	items map[string]int
}

// Add counts one more of an item and returns the new count.
func (s *Store) Add(name string) int {
	s.items[name]++
	return s.items[name]
}

func handler(s *Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, MaxBody)
		fmt.Fprintf(w, "%d", s.Add(r.URL.Query().Get("item")))
	}
}

func main() {
	s := &Store{items: map[string]int{}}
	http.ListenAndServe(":8080", handler(s))
}
