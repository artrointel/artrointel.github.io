---
title: 'WFF XML 스키마 설계 회고 2 — XSD coding convention'
excerpt: '팀의 워치페이스 명세를 일관되고 명확하게 작성하기 위한 XSD coding convention의 배경과 후속 작성 범위를 정리한 초안입니다.'
author: 'Woohyun Kim'
category: 'Platform Engineering'
tags:
  - 'Wear OS'
  - 'WFF'
  - 'XML'
  - 'XSD'
  - 'Schema Design'
language: ko
draft: true
metadata:
  robots:
    index: false
    follow: false
---

> 리뷰 중인 초안입니다. 구체적인 convention은 후속 설명을 바탕으로 작성할 예정입니다.

[시리즈 머리말](/ko/wff-xsd-design-rationale-ko)에서 소개한 XSD 도입 과정에서는 팀이 사용할 정의 방식과 coding convention도 함께 다뤘습니다.

## XSD 문법을 팀의 정의 방식으로 구체화하기

XSD 문법을 사용해 정의를 작성하는 것과, 팀의 명세를 일관되고 명확하게 작성할 규칙을 세우는 것은 함께 고려할 문제였습니다. 세미나에서도 워치페이스 요소와 구조를 어떻게 정의하고 어떤 coding convention을 적용할지 설명했습니다.

구체적인 convention과 적용 이유는 후속 리뷰에서 정리할 예정입니다. [요소·타입 설계 편의 `Part*` 회고](/ko/wff-xsd-element-design-ko)는 XML 요소 명명에 관한 사례이며, XSD coding convention 전체를 대표하는 설명으로 다루지는 않습니다.

## 다음 보강 범위

당시 적용한 규칙, 그 규칙이 필요했던 문제, 선택 이유와 적용 예시를 정리할 예정입니다. 일반적인 XSD 권장 사항을 당시 팀의 convention처럼 채워 넣지 않습니다.

## 시리즈 목차

- [머리말: 내부 위키 명세에서 XSD로](/ko/wff-xsd-design-rationale-ko)
- [1. 요소·타입 설계: Scene graph와 Part](/ko/wff-xsd-element-design-ko)
- [2. Coding convention: XSD 정의의 일관성](/ko/wff-xsd-coding-convention-ko)
- [3. Validation: 목적과 검증 경계](/ko/wff-xsd-validation-ko)
