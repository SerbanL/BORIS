#pragma once

#include "BorisLib_Config.h"
#if OPERATING_SYSTEM == OS_WIN

#include <d2d1_1.h>
#include <string>

#include <locale>
#include <codecvt>

///////////////////////////////////////////////////////////////////////////////
// GENERAL CONVERSIONS

inline std::wstring StringtoWideString(const std::string& text) 
{
	using convert_typeX = std::codecvt_utf8<wchar_t>;
	std::wstring_convert<convert_typeX, wchar_t> converterX;
	return converterX.from_bytes(text);
}

inline std::string WideStringtoString(std::wstring wstr) 
{
	using convert_typeX = std::codecvt_utf8<wchar_t>;
	std::wstring_convert<convert_typeX, wchar_t> converterX;
	return converterX.to_bytes(wstr);
}

inline WCHAR* StringtoWCHARPointer(const std::string& text) 
{
	//std::string to std::wstring conversion. To get LPCWSTR then attach .c_str() to returned std::wstring
	int len;
    int slength = (int)text.length() + 1;
    len = MultiByteToWideChar(CP_ACP, 0, text.c_str(), slength, 0, 0);
    wchar_t* buf = new wchar_t[len];
	MultiByteToWideChar(CP_ACP, 0, text.c_str(), slength, buf, len);

	return buf;
}

inline void RECTtoD2D1RECT(RECT &rc, D2D1_RECT_F &d2d1_rc) { d2d1_rc.bottom = (FLOAT)rc.bottom; d2d1_rc.top = (FLOAT)rc.top; d2d1_rc.left = (FLOAT)rc.left; d2d1_rc.right = (FLOAT)rc.right; }

inline void D2D1RECTtoRECT(D2D1_RECT_F &d2d1_rc, RECT &rc) { rc.bottom = (LONG)d2d1_rc.bottom; rc.top = (LONG)d2d1_rc.top; rc.left = (LONG)d2d1_rc.left; rc.right = (LONG)d2d1_rc.right; }

#endif