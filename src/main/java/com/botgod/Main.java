package com.botgod;

import com.botgod.word2vec.Word2VecSentimentRNN;

import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Word2VecSentimentRNN w2v = new Word2VecSentimentRNN();
        try {
            w2v.train();
            String text = "";
            while (!text.equalsIgnoreCase("end")){
                Scanner scanner = new Scanner(System.in);
                text = scanner.nextLine();
                System.out.println("Input : "+text);
                if (!text.isEmpty()) {
                    if (!text.equalsIgnoreCase("end" )) {
                        w2v.getSentiment(text);
                    } else {
                        text = "end";
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
