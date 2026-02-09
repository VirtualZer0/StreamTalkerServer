# -*- coding: utf-8 -*-
"""
Warmup phrases for torch.compile/CUDA graph optimization.

CUDA graphs capture computation for specific tensor shapes, so we warmup with
varying lengths (short/medium/long) to cover typical usage patterns.
"""

WARMUP_PHRASES = {
    "Chinese": [
        "这是测试。",
        "这是一个中等长度的预热句子，用于模型优化。",
        "这是一个较长的预热文本，它有助于确保深度学习模型的编译优化在处理较长序列文本时也能正常工作并达到最佳性能。",
    ],
    "English": [
        "Short warmup test.",
        "This is a medium length warmup sentence for the model.",
        "This is a longer warmup text that helps ensure the torch compile optimization is properly warmed up for longer sequences of text.",
    ],
    "Japanese": [
        "テストです。",
        "これはモデル最適化のための中程度の長さのウォームアップ文です。",
        "これは長めのウォームアップテキストで、ディープラーニングモデルのコンパイル最適化が長いテキストシーケンスでも適切に機能することを確認するためのものです。",
    ],
    "Korean": [
        "테스트입니다.",
        "이것은 모델 최적화를 위한 중간 길이의 워밍업 문장입니다.",
        "이것은 긴 워밍업 텍스트로, 딥러닝 모델의 컴파일 최적화가 긴 텍스트 시퀀스에서도 제대로 작동하는지 확인하는 데 도움이 됩니다.",
    ],
    "German": [
        "Kurzer Aufwärmtest.",
        "Dies ist ein mittelanger Aufwärmsatz für die Modelloptimierung.",
        "Dies ist ein längerer Aufwärmtext, der sicherstellt, dass die Kompilierungsoptimierung des Deep-Learning-Modells auch bei längeren Textsequenzen ordnungsgemäß funktioniert.",
    ],
    "French": [
        "Test de chauffe.",
        "Ceci est une phrase de préchauffage de longueur moyenne pour l'optimisation du modèle.",
        "Ceci est un texte de préchauffage plus long qui aide à s'assurer que l'optimisation de compilation du modèle fonctionne correctement pour les séquences de texte plus longues.",
    ],
    "Russian": [
        "Короткий тест.",
        "Это предложение средней длины для прогрева модели и оптимизации.",
        "Это более длинный текст для прогрева, который помогает убедиться что оптимизация компиляции модели глубокого обучения правильно работает для длинных текстовых последовательностей.",
    ],
    "Portuguese": [
        "Teste curto.",
        "Esta é uma frase de aquecimento de comprimento médio para otimização do modelo.",
        "Este é um texto de aquecimento mais longo que ajuda a garantir que a otimização de compilação do modelo de aprendizado profundo funcione corretamente para sequências de texto mais longas.",
    ],
    "Spanish": [
        "Prueba corta.",
        "Esta es una oración de calentamiento de longitud media para la optimización del modelo.",
        "Este es un texto de calentamiento más largo que ayuda a asegurar que la optimización de compilación del modelo de aprendizaje profundo funcione correctamente para secuencias de texto más largas.",
    ],
    "Italian": [
        "Test breve.",
        "Questa è una frase di riscaldamento di media lunghezza per l'ottimizzazione del modello.",
        "Questo è un testo di riscaldamento più lungo che aiuta a garantire che l'ottimizzazione della compilazione del modello di deep learning funzioni correttamente per sequenze di testo più lunghe.",
    ],
}

DEFAULT_WARMUP_LANGUAGE = "English"


def get_warmup_phrases(language: str = None) -> tuple[list[str], str]:
    """
    Get warmup phrases for the specified language.
    
    Args:
        language: Language name (e.g., "English", "Russian"). 
                  If None, uses DEFAULT_WARMUP_LANGUAGE.
    
    Returns:
        Tuple of (phrases list, resolved language name)
    """
    if language is None:
        language = DEFAULT_WARMUP_LANGUAGE
    
    phrases = WARMUP_PHRASES.get(language)
    if phrases:
        return phrases, language
    
    # Fallback to English
    return WARMUP_PHRASES.get("English", ["Warmup test."]), "English"
