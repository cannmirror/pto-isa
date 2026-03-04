// --------------------------------------------------------------------------------
// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// --------------------------------------------------------------------------------

/**
 * Navigation collapse control
 * 
 * This script collapses all second-level navigation items by default,
 * showing only the first-level navigation items.
 * Works with the existing +/- buttons in the ReadTheDocs theme.
 */
(function() {
    'use strict';

    function collapseSecondLevelNav() {
        // 查找所有一级导航项
        const firstLevelItems = document.querySelectorAll('.toctree-l1');
        
        firstLevelItems.forEach(item => {
            // 检查是否有子项（ul 元素）
            const subMenu = item.querySelector('ul');
            if (!subMenu) return;
            
            // 如果当前项不是激活状态，则收起子菜单
            if (!item.classList.contains('current')) {
                // 使用 jQuery 的方式（ReadTheDocs 主题使用 jQuery）
                if (typeof $ !== 'undefined') {
                    $(subMenu).hide();
                } else {
                    // 如果没有 jQuery，直接设置样式
                    subMenu.style.display = 'none';
                }
            }
        });
        
        // 更新所有 +/- 按钮的显示
        updateExpandButtons();
        
        // 监听按钮点击事件，点击后更新按钮显示
        const expandButtons = document.querySelectorAll('.toctree-expand');
        expandButtons.forEach(button => {
            button.addEventListener('click', function() {
                // 延迟更新，等待子菜单展开/收起动画完成
                setTimeout(updateExpandButtons, 50);
            });
        });
    }
    
    // 更新所有 +/- 按钮的显示
    function updateExpandButtons() {
        const firstLevelItems = document.querySelectorAll('.toctree-l1');
        
        firstLevelItems.forEach(item => {
            const subMenu = item.querySelector('ul');
            const button = item.querySelector('button.toctree-expand');
            
            if (!subMenu || !button) return;
            
            // 检查子菜单是否可见
            const isVisible = subMenu.style.display !== 'none' && 
                             window.getComputedStyle(subMenu).display !== 'none';
            
            // 根据可见性设置按钮的 data 属性，CSS 可以使用这个属性
            if (isVisible) {
                button.setAttribute('data-expanded', 'true');
            } else {
                button.setAttribute('data-expanded', 'false');
            }
        });
    }

    // 初始化
    function init() {
        // 延迟执行，确保主题的 JavaScript 已经加载
        setTimeout(collapseSecondLevelNav, 200);
    }

    // 页面加载完成后初始化
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();

